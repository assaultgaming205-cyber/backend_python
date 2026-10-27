import asyncio
import io
import logging
import queue
import wave
from typing import Optional

import httpx
try:
    import riva.client
except ImportError:
    riva = None
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from ..config import settings

logger = logging.getLogger("stt")
router = APIRouter(tags=["stt"])

STT_LANGUAGES = [
    {"code": c, "name": n}
    for c, n in [("auto", "Auto"), ("hi", "Hindi"), ("en", "English"), ("mr", "Marathi"), ("gu", "Gujarati"),
                 ("ta", "Tamil"), ("te", "Telugu"), ("bn", "Bengali"), ("kn", "Kannada"), ("ml", "Malayalam"),
                 ("pa", "Punjabi"), ("ur", "Urdu")]
]
FUNCTIONS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/functions"
PARAKEET_NAME = "ai-parakeet-1_1b-rnnt-multilingual-asr"
PARAKEET_KNOWN_ID = "71203149-d3b7-4460-8231-1be2543a1fca"  # fallback if the NVCF lookup fails

_function_id: Optional[str] = None


def _api_key() -> str:
    return settings.NVIDIA_ASR_API_KEY or settings.NVIDIA_API_KEY


def _available() -> bool:
    if settings.SARVAM_STT_ENABLED and settings.SARVAM_API_KEY:
        return True
    return bool(settings.STT_ENABLED and _api_key() and riva is not None)


def _language_code(lang: Optional[str]) -> str:
    lang = (lang or "auto").lower().split("-")[0].split("_")[0]
    if lang in ("auto", "hi", "hinglish", ""):
        return "hi-IN"  # Indian users mostly speak Hindi/Hinglish
    return "en-US" if lang == "en" else "multi"


def _resolve_function_id() -> str:
    global _function_id
    if _function_id:
        return _function_id
    try:
        r = httpx.get(FUNCTIONS_URL, headers={"authorization": f"Bearer {_api_key()}"}, timeout=8)
        r.raise_for_status()
        fns = [f for f in r.json().get("functions", []) if f.get("name") == PARAKEET_NAME]
        fns.sort(key=lambda f: f.get("status") != "ACTIVE")
        if fns:
            _function_id = fns[0]["id"]
            return _function_id
    except Exception as e:
        logger.warning(f"NVCF function lookup failed ({e}); using known id")
    _function_id = PARAKEET_KNOWN_ID
    return _function_id


def _asr_service():
    if riva is None:
        raise RuntimeError("nvidia-riva-client is not installed. Install with: pip install nvidia-riva-client")
    auth = riva.client.Auth(
        None, True, settings.NVIDIA_ASR_GRPC_URL,
        [["function-id", _resolve_function_id()], ["authorization", f"Bearer {_api_key()}"]],
    )
    return riva.client.ASRService(auth)


def _recog_config(sample_rate: int, language_code: str):
    if riva is None:
        raise RuntimeError("nvidia-riva-client is not installed. Install with: pip install nvidia-riva-client")
    return riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        max_alternatives=1,
        audio_channel_count=1,
        enable_automatic_punctuation=True,
    )


def _join(finals, interim) -> str:
    return " ".join(p for p in [*finals, interim] if p).strip()


@router.get("/stt/options")
@router.get("/api/stt/options")
async def stt_options():
    engines = {"parakeet": {"model": "parakeet-1.1b-rnnt-multilingual", "mode": "streaming"}}
    if settings.SARVAM_STT_ENABLED:
        engines["sarvam"] = {"model": settings.SARVAM_STT_MODEL, "mode": "realtime_streaming"}
    return {
        "available": _available(),
        "provider": "sarvam-realtime" if settings.SARVAM_STT_ENABLED else "nvidia-riva",
        "streaming": True,
        "sampleRate": 16000,
        "encoding": "pcm16le",
        "maxSeconds": 60,
        "defaultLanguage": settings.STT_DEFAULT_LANGUAGE,
        "engines": engines,
        "languages": STT_LANGUAGES,
        "streamPath": "/api/stt/stream",
    }


def _to_pcm(audio: bytes, default_rate: int):
    if audio[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio)) as w:
            return w.readframes(w.getnframes()), w.getframerate()
    return audio, default_rate


@router.post("/stt/transcribe")
@router.post("/api/stt/transcribe")
async def stt_transcribe(request: Request, language: Optional[str] = Query(None), sampleRate: int = Query(16000)):
    body = await request.body()
    if not body:
        raise HTTPException(400, {"code": "empty_audio", "message": "Send audio/wav or PCM16."})
    if not _available():
        raise HTTPException(503, {"code": "stt_unconfigured", "message": "NVIDIA_API_KEY is not set."})
    try:
        pcm, rate = _to_pcm(body, sampleRate)
    except Exception:
        raise HTTPException(400, {"code": "bad_audio", "message": "Could not read WAV audio."})

    def run() -> str:
        res = _asr_service().offline_recognize(pcm, _recog_config(rate, _language_code(language)))
        return " ".join(r.alternatives[0].transcript.strip() for r in res.results if r.alternatives).strip()

    try:
        text = await asyncio.to_thread(run)
    except Exception as e:
        raise HTTPException(502, {"code": "stt_failed", "message": f"Speech recognition failed: {e}"})
    return {"text": text, "language": language or "auto", "engine": "parakeet"}


@router.websocket("/api/stt/stream")
@router.websocket("/stt/stream")
async def stt_stream(ws: WebSocket):
    """One utterance per connection: start -> PCM16 frames -> stop; server sends ready/partial/final/error/status."""
    await ws.accept()
    logger.info("STT WebSocket client connected")
    if not _available():
        logger.warning("STT requested but STT is not configured; closing.")
        await ws.send_json({"type": "error", "code": "stt_unavailable", "message": "Speech recognition is not configured."})
        await ws.close()
        return

    import time
    from ..ai.stt_service import FailoverSttSession
    session: Optional[FailoverSttSession] = None
    sender_task: Optional[asyncio.Task] = None
    started = False
    start_time_mono = time.monotonic()
    early_audio_buffer: list[bytes] = []

    async def sender():
        last = ""
        try:
            assert session is not None
            async for ev in session.receive_events():
                if ev.kind == "partial":
                    if ev.text and ev.text != last:
                        last = ev.text
                        await ws.send_json({"type": "partial", "text": ev.text})
                elif ev.kind == "final":
                    latency_ms = int((time.monotonic() - start_time_mono) * 1000)
                    logger.info(f"STT final emitted: chars={len(ev.text)}, engine={ev.engine}, lang={ev.language}, latency={latency_ms}ms")
                    await ws.send_json({
                        "type": "final",
                        "text": ev.text,
                        "language": ev.language,
                        "engine": ev.engine,
                        "latencyMs": latency_ms,
                    })
                    return
                elif ev.kind == "status":
                    logger.info(f"STT provider status: {ev.text}")
                    await ws.send_json({"type": "status", "message": ev.text, "engine": ev.engine})
                elif ev.kind == "speech_end":
                    logger.info(f"STT speech_end detected by {ev.engine}")
                    await ws.send_json({"type": "speech_end", "engine": ev.engine})
                elif ev.kind == "error":
                    logger.warning(f"STT error event: code={ev.error_code}, msg={ev.text}")
                    await ws.send_json({"type": "error", "code": ev.error_code or "stt_failed", "message": ev.text})
                    return
        except Exception as e:
            logger.warning(f"Error in STT sender task: {e}")

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                logger.info("STT client disconnected WebSocket")
                break
            if msg.get("bytes"):
                if started and session:
                    await session.send_audio(msg["bytes"])
                else:
                    if len(early_audio_buffer) < 100:
                        early_audio_buffer.append(msg["bytes"])
                continue
            text = msg.get("text")
            if not text:
                continue
            import json
            data = json.loads(text)
            kind = data.get("type")
            if kind == "start" and not started:
                started = True
                rate = max(8000, min(48000, int(data.get("sampleRate") or 16000)))
                lang = data.get("language") or "auto"
                start_time_mono = time.monotonic()
                session = FailoverSttSession(sarvam_enabled=bool(settings.SARVAM_STT_ENABLED))
                await session.start(rate, lang)
                sender_task = asyncio.create_task(sender())
                if early_audio_buffer:
                    for early_chunk in early_audio_buffer:
                        await session.send_audio(early_chunk)
                    early_audio_buffer.clear()
                logger.info(f"STT stream started: rate={rate}, lang={lang}, provider={session.active_provider}")
                await ws.send_json({"type": "ready", "sampleRate": rate})
            elif kind == "stop":
                logger.info("STT stream stop requested by client; waiting for final transcript")
                if session:
                    await session.stop()
                if sender_task:
                    try:
                        await asyncio.wait_for(sender_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("STT sender timed out waiting for final transcript.")
                break
            elif kind == "cancel":
                logger.info("STT stream cancel requested by client")
                if session:
                    await session.cancel()
                if sender_task:
                    sender_task.cancel()
                break
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("STT WebSocket disconnected")
    finally:
        normal = "yes" if (session and session._final_emitted) else "no"
        logger.info(f"STT stream cleanup finished: final_delivered={normal}")
        if session:
            await session.cancel()
        if sender_task and not sender_task.done():
            sender_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
