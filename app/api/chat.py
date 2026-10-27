import asyncio
import time
import json
import uuid
import logging
from typing import List, Optional, Literal, Any, Dict
from fastapi import APIRouter, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..schemas.chat import ChatResponse, ChatMessage, ChatMetadata
from ..ai.ai_service import AiMessage
from ..ai.tts_streaming_service import StreamingTtsSession
from ..agent.agent_service import AgentService

logger = logging.getLogger("chat_controller")
router = APIRouter(tags=["chat"])
agent_service = AgentService()


class ChatMessageDto(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    selectedOfferId: Optional[str] = None
    clarificationAnswer: Optional[Dict[str, Any]] = None


class ChatRequestDto(BaseModel):
    messages: List[ChatMessageDto]
    sessionId: Optional[str] = None
    character: Optional[Literal["meera", "kabir"]] = "meera"
    modelProvider: Optional[Literal["nvidia", "local"]] = "nvidia"
    model: Optional[str] = None
    interactiveMode: Optional[bool] = False
    presenceCheck: Optional[bool] = False
    clarificationAnswer: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(dto: ChatRequestDto):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"Incoming chat turn (session={dto.sessionId}, character={dto.character}, model={dto.model})")

    history = [
        AiMessage(
            role=m.role,
            content=m.content,
            selected_offer_id=m.selectedOfferId,
            clarification_answer=m.clarificationAnswer,
        )
        for m in dto.messages
    ]

    result = await agent_service.run(
        history=history,
        session_id=dto.sessionId,
        model_provider=dto.modelProvider,
        interactive_mode=dto.interactiveMode or False,
        character=dto.character or "meera",
        model=dto.model,
        clarification_answer=dto.clarificationAnswer,
    )

    duration_ms = int((time.time() - start_time) * 1000)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=result.content),
        narration=result.narration,
        toolCalls=result.toolCalls,
        results=result.results,
        metadata=ChatMetadata(
            requestId=request_id,
            model=result.model,
            durationMs=duration_ms,
            sessionId=result.sessionId,
        ),
    )


@router.post("/chat/stream")
@router.post("/api/chat/stream")
async def chat_stream_endpoint(dto: ChatRequestDto, request: Request):
    """Server-Sent Events (SSE) streaming endpoint for conversational travel assistant turns."""
    start_time = time.time()
    turn_id = str(uuid.uuid4())

    async def event_generator():
        start_time_mono = time.monotonic()
        first_sentence_mono: Optional[float] = None
        first_sentence_ms: Optional[int] = None
        first_audio_mono: Optional[float] = None
        tts_first_chunk_ms: Optional[int] = None
        events: asyncio.Queue = asyncio.Queue()
        sentence_counter = 0

        # Streaming TTS initialization
        tts_enabled = bool(getattr(settings, "STREAMING_TTS_ENABLED", False))
        tts_session: Optional[StreamingTtsSession] = None
        tts_pump_task: Optional[asyncio.Task] = None

        if tts_enabled:
            tts_session = StreamingTtsSession(
                turn_id=turn_id,
                character=dto.character or "meera",
            )
            connected = await tts_session.connect()
            if connected:
                async def pump_audio():
                    nonlocal first_audio_mono, tts_first_chunk_ms
                    try:
                        async for chunk in tts_session.receive_chunks():
                            if first_audio_mono is None and chunk.audio:
                                first_audio_mono = time.monotonic()
                                tts_first_chunk_ms = int((first_audio_mono - start_time_mono) * 1000)
                            events.put_nowait({
                                "type": "audio",
                                "turnId": turn_id,
                                "sentenceIndex": chunk.sentence_index,
                                "chunkIndex": chunk.chunk_index,
                                "format": chunk.format,
                                "audio": chunk.audio,
                                "isLast": chunk.is_last,
                            })
                    except Exception as e:
                        logger.error(f"Error pumping streaming TTS audio: {e}")

                tts_pump_task = asyncio.create_task(pump_audio())
            else:
                events.put_nowait({
                    "type": "status",
                    "turnId": turn_id,
                    "statusText": "Voice audio is currently unavailable; continuing with text response.",
                })
                tts_session = None

        tts_mode = "streaming" if (tts_enabled and tts_session and tts_session.is_connected) else "rest"

        # 1. Announce turn start and initial status
        yield f"data: {json.dumps({'type': 'turn_started', 'turnId': turn_id, 'sessionId': dto.sessionId, 'ttsMode': tts_mode})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'turnId': turn_id, 'statusText': 'Checking your request…'})}\n\n"

        history = [
            AiMessage(
                role=m.role,
                content=m.content,
                selected_offer_id=m.selectedOfferId,
                clarification_answer=m.clarificationAnswer,
            )
            for m in dto.messages
        ]

        def handle_status(s: str):
            events.put_nowait({"type": "status", "turnId": turn_id, "statusText": s})

        def handle_clarification(card: Dict[str, Any]):
            events.put_nowait(card)

        def handle_card(card_type: str, data: Any):
            card_payload = {"type": "card", "turnId": turn_id, "cardType": card_type}
            if card_type == "flight":
                card_payload["flights"] = [f.model_dump() for f in data]
                if data and hasattr(data[0], "searchContextId") and data[0].searchContextId:
                    card_payload["searchContextId"] = data[0].searchContextId
            elif card_type == "train":
                card_payload["trains"] = [t.model_dump() for t in data]
            elif card_type == "hotel":
                card_payload["hotels"] = [h.model_dump() for h in data]
            events.put_nowait(card_payload)

        def handle_sentence(s: str):
            nonlocal sentence_counter, first_sentence_mono, first_sentence_ms
            if first_sentence_mono is None:
                first_sentence_mono = time.monotonic()
                first_sentence_ms = int((first_sentence_mono - start_time_mono) * 1000)
            events.put_nowait({
                "type": "sentence",
                "turnId": turn_id,
                "text": s,
                "delta": s,
                "sentenceIndex": sentence_counter,
            })
            if tts_session and tts_session.is_connected:
                asyncio.create_task(tts_session.send_sentence(s, sentence_counter))
            sentence_counter += 1

        def handle_results(r):
            events.put_nowait({
                "type": "results",
                "turnId": turn_id,
                "results": r.model_dump(),
                "searchContextId": getattr(r, "searchContextId", None),
            })

        task = asyncio.create_task(
            agent_service.run(
                history=history,
                session_id=dto.sessionId,
                model_provider=dto.modelProvider,
                interactive_mode=dto.interactiveMode or False,
                character=dto.character or "meera",
                model=dto.model,
                on_status=handle_status,
                on_results=handle_results,
                on_card=handle_card,
                on_sentence=handle_sentence,
                on_clarification=handle_clarification,
                clarification_answer=dto.clarificationAnswer,
            )
        )

        idle = 0
        cancelled = False
        try:
            while not task.done() or not events.empty():
                if await request.is_disconnected():
                    logger.info(f"Client disconnected for turn {turn_id}. Cancelling agent and TTS tasks.")
                    if tts_session:
                        await tts_session.close()
                    if tts_pump_task:
                        tts_pump_task.cancel()
                    task.cancel()
                    cancelled = True
                    yield f"data: {json.dumps({'type': 'turn_cancelled', 'turnId': turn_id, 'reason': 'client_disconnected'})}\n\n"
                    break

                try:
                    ev = await asyncio.wait_for(events.get(), timeout=0.25)
                    idle = 0
                except asyncio.TimeoutError:
                    idle += 1
                    if idle % 20:  # heartbeat every ~5s keeps connection alive
                        continue
                    ev = {"type": "heartbeat", "turnId": turn_id}

                yield f"data: {json.dumps(ev)}\n\n"

            if cancelled:
                return

            # Flush TTS streaming session and drain remaining audio chunks
            if tts_session and tts_session.is_connected:
                await tts_session.flush()
                if tts_pump_task:
                    try:
                        await asyncio.wait_for(tts_pump_task, timeout=3.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

            try:
                result = task.result()
            except asyncio.CancelledError:
                logger.info(f"Agent task for turn {turn_id} was cancelled.")
                yield f"data: {json.dumps({'type': 'turn_cancelled', 'turnId': turn_id, 'reason': 'task_cancelled'})}\n\n"
                return
            except Exception as e:
                logger.error(f"Chat stream failed for turn {turn_id}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'turnId': turn_id, 'errorMessage': 'This turn could not complete. Please try again.'})}\n\n"
                return

            duration_ms = int((time.time() - start_time) * 1000)
            narration_data = [n.model_dump() for n in result.narration]

            # In legacy non-streaming mode, emit legacy chunks and narration
            if not getattr(settings, "LLM_TOKEN_STREAMING_ENABLED", False):
                yield f"data: {json.dumps({'type': 'results', 'turnId': turn_id, 'results': result.results.model_dump()})}\n\n"
                yield f"data: {json.dumps({'type': 'narration', 'turnId': turn_id, 'sections': narration_data, 'content': result.content})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'turnId': turn_id, 'delta': result.content})}\n\n"

            telemetry_data = {
                "turnId": turn_id,
                "ttft_ms": result.telemetry.get("ttft_ms") if (result.telemetry and result.telemetry.get("ttft_ms") is not None) else None,
                "first_sentence_ms": first_sentence_ms,
                "tts_first_chunk_ms": tts_first_chunk_ms,
                "suppliers_ms": result.telemetry.get("suppliers_ms", {}) if result.telemetry else {},
                "total_backend_ms": duration_ms,
            }

            done_payload = {
                "type": "done",
                "turnId": turn_id,
                "content": result.content,
                "narration": narration_data,
                "sessionId": result.sessionId,
                "searchContextId": getattr(result.results, "searchContextId", None),
                "model": result.model,
                "durationMs": duration_ms,
                "toolCalls": [t.model_dump() for t in result.toolCalls],
                "results": result.results.model_dump(),
                "telemetry": telemetry_data,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        finally:
            if tts_session:
                await tts_session.close()
            if tts_pump_task and not tts_pump_task.done():
                tts_pump_task.cancel()
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class RetryDateRequestDto(BaseModel):
    sessionId: str
    date: str
    cabin: Optional[str] = None
    searchContextId: Optional[str] = None


@router.post("/chat/retry_date")
@router.post("/api/chat/retry_date")
async def retry_date_endpoint(dto: RetryDateRequestDto):
    res = await agent_service.retry_date(
        dto.sessionId,
        dto.date,
        search_context_id=dto.searchContextId,
        cabin=dto.cabin,
    )
    return res


