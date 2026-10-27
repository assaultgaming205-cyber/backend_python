import asyncio
import json
import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator, List

from fastapi import Request
from starlette.datastructures import Headers

from app.config import settings
from app.ai.tts_streaming_service import (
    StreamingTtsSession,
    sanitize_for_tts,
    TtsAudioChunk,
)
from app.agent.agent_service import AgentService, AgentResult
from app.schemas.chat import ChatResults
from app.api.chat import (
    chat_stream_endpoint,
    chat_endpoint,
    ChatRequestDto,
    ChatMessageDto,
    agent_service,
)


class MockWebSocket:
    """Mock WebSocket matching websockets client contract."""

    def __init__(self, incoming_messages: List[str] = None):
        self.sent_messages: List[str] = []
        self.incoming_messages = list(incoming_messages or [])
        self.closed = False
        self.close_code = None

    async def send(self, data: str):
        if self.closed:
            raise ConnectionResetError("Cannot send on closed socket")
        self.sent_messages.append(data)

    async def recv(self) -> str:
        if self.closed:
            raise ConnectionResetError("Socket closed")
        if not self.incoming_messages:
            # Block or wait briefly then raise
            await asyncio.sleep(0.01)
            raise ConnectionResetError("No more messages")
        return self.incoming_messages.pop(0)

    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.recv()
        except ConnectionResetError:
            raise StopAsyncIteration


class TestStreamingTtsService(unittest.IsolatedAsyncioTestCase):
    """Deterministic tests for StreamingTtsSession without paid APIs."""

    def setUp(self):
        self.orig_key = settings.SARVAM_API_KEY
        settings.SARVAM_API_KEY = "dummy-test-key-12345"

    def tearDown(self):
        settings.SARVAM_API_KEY = self.orig_key

    async def test_config_sent_before_text(self):
        mock_ws = MockWebSocket(incoming_messages=[
            json.dumps({"type": "event", "data": {"event_type": "ready"}}),
            json.dumps({"type": "audio", "data": {"audio": base64.b64encode(b"chunk1").decode()}}),
            json.dumps({"type": "event", "data": {"event_type": "final"}}),
        ])

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            session = StreamingTtsSession(turn_id="turn-1", character="meera", model="bulbul:v4-flash")
            connected = await session.connect()
            self.assertTrue(connected)

            # First message sent MUST be config frame
            self.assertGreater(len(mock_ws.sent_messages), 0)
            first_msg = json.loads(mock_ws.sent_messages[0])
            self.assertEqual(first_msg.get("type"), "config")
            config_data = first_msg.get("data", {})
            self.assertEqual(config_data.get("model"), "bulbul:v4-flash")
            self.assertIn("hi", config_data.get("speaker", ""))
            self.assertEqual(config_data.get("output_audio_codec"), "mp3")

            # Now send text
            await session.send_sentence("IndiGo flight 6E-204 is available.", sentence_index=0)
            self.assertEqual(len(mock_ws.sent_messages), 2)
            second_msg = json.loads(mock_ws.sent_messages[1])
            self.assertEqual(second_msg.get("type"), "text")
            self.assertIn("IndiGo flight 6E-204 is available.", second_msg.get("data", {}).get("text", ""))

            await session.close()
            self.assertTrue(mock_ws.closed)

    async def test_multiple_sentences_and_flush(self):
        mock_ws = MockWebSocket(incoming_messages=[])
        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            session = StreamingTtsSession(turn_id="turn-2")
            await session.connect()
            await session.send_sentence("Sentence one.", sentence_index=0)
            await session.send_sentence("Sentence two.", sentence_index=1)
            await session.flush()

            # Expect: config, text 1, text 2, flush
            self.assertEqual(len(mock_ws.sent_messages), 4)
            self.assertEqual(json.loads(mock_ws.sent_messages[0])["type"], "config")
            self.assertEqual(json.loads(mock_ws.sent_messages[1])["type"], "text")
            self.assertEqual(json.loads(mock_ws.sent_messages[2])["type"], "text")
            self.assertEqual(json.loads(mock_ws.sent_messages[3])["type"], "flush")
            await session.close()

    async def test_audio_chunks_delivered_in_order(self):
        b64_1 = base64.b64encode(b"audio-part-1").decode()
        b64_2 = base64.b64encode(b"audio-part-2").decode()

        mock_ws = MockWebSocket(incoming_messages=[
            json.dumps({"type": "audio", "data": {"audio": b64_1}}),
            json.dumps({"type": "audio", "data": {"audio": b64_2}}),
            json.dumps({"type": "event", "data": {"event_type": "final"}}),
        ])

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            session = StreamingTtsSession(turn_id="turn-3")
            await session.connect()
            await session.send_sentence("Flight details.", sentence_index=0)

            chunks: List[TtsAudioChunk] = []
            async for chunk in session.receive_chunks():
                chunks.append(chunk)
                if chunk.is_last:
                    break

            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].chunk_index, 0)
            self.assertEqual(chunks[0].audio, b64_1)
            self.assertFalse(chunks[0].is_last)

            self.assertEqual(chunks[1].chunk_index, 1)
            self.assertEqual(chunks[1].audio, b64_2)
            self.assertFalse(chunks[1].is_last)

            self.assertTrue(chunks[2].is_last)

            await session.close()

    async def test_v4_flash_failure_never_falls_back_to_v3(self):
        """If v4-flash fails to connect, session fails gracefully and never requests bulbul:v3."""
        requested_urls = []

        async def fake_connect(url, **kwargs):
            requested_urls.append(url)
            raise ConnectionError("Model bulbul:v4-flash unavailable on endpoint")

        with patch("websockets.connect", side_effect=fake_connect):
            session = StreamingTtsSession(turn_id="turn-fb", model="bulbul:v4-flash")
            connected = await session.connect()
            self.assertFalse(connected)
            self.assertFalse(session.is_connected)
            # Verify bulbul:v4-flash was attempted, and bulbul:v3 was NEVER attempted
            self.assertEqual(len(requested_urls), 1)
            self.assertIn("model=bulbul:v4-flash", requested_urls[0])
            self.assertNotIn("v3", requested_urls[0])
            self.assertEqual(session.model, "bulbul:v4-flash")
            await session.close()

    async def test_persona_ids_for_meera_and_kabir(self):
        """WebSocket config frame specifies correct Bulbul V4 Flash personas."""
        mock_ws_meera = MockWebSocket(incoming_messages=[])
        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws_meera)):
            session = StreamingTtsSession(turn_id="turn-p1", character="meera")
            await session.connect()
            config = json.loads(mock_ws_meera.sent_messages[0])["data"]
            self.assertEqual(config["model"], "bulbul:v4-flash")
            self.assertEqual(config["speaker"], "priya_hi_recovery")
            await session.close()

        mock_ws_kabir = MockWebSocket(incoming_messages=[])
        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws_kabir)):
            session = StreamingTtsSession(turn_id="turn-p2", character="kabir")
            await session.connect()
            config = json.loads(mock_ws_kabir.sent_messages[0])["data"]
            self.assertEqual(config["model"], "bulbul:v4-flash")
            self.assertEqual(config["speaker"], "kabir_hi_conversational")
            await session.close()

    async def test_rest_tts_service_selects_v4_flash_and_never_v3(self):
        """REST synthesis explicitly specifies bulbul:v4-flash and never calls bulbul:v3."""
        from app.ai.tts_service import TtsService
        service = TtsService()
        service.api_key = "dummy-key"

        captured_requests = []

        class MockResponse:
            status_code = 200
            text = "ok"
            def json(self):
                return {"audios": [base64.b64encode(b"RIFFdummydata").decode()]}

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None, headers=None):
                captured_requests.append({"url": url, "json": json, "headers": headers})
                return MockResponse()

        with patch("httpx.AsyncClient", side_effect=MockAsyncClient):
            # Test Meera (female persona)
            audio = await service.synthesize("Delhi se Mumbai jana hai", character="meera")
            self.assertTrue(len(audio) > 0)
            self.assertEqual(len(captured_requests), 1)
            req = captured_requests[0]["json"]
            self.assertEqual(req["model"], "bulbul:v4-flash")
            self.assertEqual(req["speaker"], "priya_hi_recovery")
            self.assertNotIn("v3", req["model"])

            # Test Kabir (male persona)
            captured_requests.clear()
            service.cache.clear()
            audio_kabir = await service.synthesize("Ticket book karo", character="kabir")
            self.assertTrue(len(audio_kabir) > 0)
            self.assertEqual(len(captured_requests), 1)
            req_kabir = captured_requests[0]["json"]
            self.assertEqual(req_kabir["model"], "bulbul:v4-flash")
            self.assertEqual(req_kabir["speaker"], "kabir_hi_conversational")
            self.assertNotIn("v3", req_kabir["model"])

    async def test_rest_tts_failure_preserves_text_and_never_falls_back_to_v3(self):
        """When provider fails, REST returns empty audio, never attempts V3 retry."""
        from app.ai.tts_service import TtsService
        service = TtsService()
        service.api_key = "dummy-key"

        captured_requests = []

        class MockFailResponse:
            status_code = 503
            text = "Service Unavailable"

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None, headers=None):
                captured_requests.append({"url": url, "json": json})
                return MockFailResponse()

        with patch("httpx.AsyncClient", side_effect=MockAsyncClient):
            audio = await service.synthesize("Hello world", character="meera")
            self.assertEqual(audio, b"")
            self.assertEqual(len(captured_requests), 1)
            self.assertEqual(captured_requests[0]["json"]["model"], "bulbul:v4-flash")
            self.assertNotIn("v3", captured_requests[0]["json"]["model"])

    async def test_secret_api_key_not_leaked_in_logs(self):
        """API key must never appear in raw logged messages or string representations."""
        session = StreamingTtsSession(turn_id="turn-sec")
        rep = str(session)
        self.assertNotIn("dummy-test-key-12345", rep)

    async def test_malformed_incoming_json_handled_safely(self):
        mock_ws = MockWebSocket(incoming_messages=[
            "NOT_JSON_AT_ALL",
            json.dumps({"type": "audio", "data": {}}),  # Missing 'audio'
            json.dumps({"type": "audio", "data": {"audio": base64.b64encode(b"valid").decode()}}),
            json.dumps({"type": "event", "data": {"event_type": "final"}}),
        ])

        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            session = StreamingTtsSession(turn_id="turn-err")
            await session.connect()
            await session.send_sentence("Testing errors.", sentence_index=0)

            received = []
            async for chunk in session.receive_chunks():
                received.append(chunk)
                if chunk.is_last:
                    break

            # Malformed frames were ignored, valid audio chunk and final marker were yielded
            self.assertEqual(len(received), 2)
            self.assertEqual(received[0].audio, base64.b64encode(b"valid").decode())
            self.assertTrue(received[1].is_last)
            await session.close()

    async def test_cancellation_closes_upstream_socket(self):
        mock_ws = MockWebSocket(incoming_messages=[])
        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            session = StreamingTtsSession(turn_id="turn-cancel")
            await session.connect()
            self.assertFalse(mock_ws.closed)
            await session.close()
            self.assertTrue(mock_ws.closed)


class TestRomanizedHinglishSanitizer(unittest.TestCase):
    """Part 6: Romanized Hinglish sanitization requirements."""

    def test_devanagari_transliterated_to_roman(self):
        text = "यह flight बहुत अच्छी है।"
        sanitized = sanitize_for_tts(text)
        # Should not contain Devanagari characters
        for char in sanitized:
            self.assertNotIn("\u0900" <= char <= "\u097f", [True], f"Devanagari char {char} found in {sanitized}")
        self.assertIn("flight", sanitized)

    def test_markdown_and_emojis_stripped(self):
        text = "✈️ **IndiGo** flight 6E-204 *departs* at 14:30! 🎉"
        sanitized = sanitize_for_tts(text)
        self.assertNotIn("✈️", sanitized)
        self.assertNotIn("🎉", sanitized)
        self.assertNotIn("**", sanitized)
        self.assertNotIn("*", sanitized)
        self.assertIn("IndiGo flight 6E-204 departs at 14:30", sanitized)

    def test_empty_text_returns_empty(self):
        self.assertEqual(sanitize_for_tts(""), "")
        self.assertEqual(sanitize_for_tts("   "), "")


class TestChatStreamTtsIntegration(unittest.IsolatedAsyncioTestCase):
    """Part 3 & Part 7E: End-to-end audio SSE streaming in /api/chat/stream."""

    def setUp(self):
        self.orig_stream_tts = settings.STREAMING_TTS_ENABLED
        self.orig_barge = settings.BARGE_IN_ENABLED
        self.orig_key = settings.SARVAM_API_KEY
        settings.SARVAM_API_KEY = "test-key"

    def tearDown(self):
        settings.STREAMING_TTS_ENABLED = self.orig_stream_tts
        settings.BARGE_IN_ENABLED = self.orig_barge
        settings.SARVAM_API_KEY = self.orig_key

    async def test_audio_events_emitted_when_streaming_tts_enabled(self):
        settings.STREAMING_TTS_ENABLED = True

        async def fake_agent_run(*args, **kwargs):
            if kwargs.get("on_sentence"):
                kwargs["on_sentence"]("IndiGo flight 6E-204 is available. Fares start at ₹4,500.")
            return AgentResult(
                content="IndiGo flight 6E-204 is available. Fares start at ₹4,500.",
                model="mock-model",
                results=ChatResults()
            )

        b64_audio = base64.b64encode(b"mp3-audio-bytes").decode()
        mock_ws = MockWebSocket(incoming_messages=[
            json.dumps({"type": "audio", "data": {"audio": b64_audio}}),
            json.dumps({"type": "event", "data": {"event_type": "final"}}),
        ])

        payload = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Show flights to Mumbai")],
            interactiveMode=True,
            character="meera"
        )
        fake_req = MagicMock(spec=Request)
        fake_req.is_disconnected = AsyncMock(return_value=False)

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", True):
            with patch.object(agent_service, "run", side_effect=fake_agent_run):
                with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
                    response = await chat_stream_endpoint(payload, fake_req)
                    events = []
                    async for chunk in response.body_iterator:
                        for line in chunk.split("\n\n"):
                            line = line.strip()
                            if line.startswith("data: "):
                                events.append(json.loads(line[6:]))

                    types = [e["type"] for e in events]
                    self.assertIn("turn_started", types)
                    self.assertIn("sentence", types)
                    self.assertIn("audio", types)

                    audio_events = [e for e in events if e["type"] == "audio"]
                    self.assertGreater(len(audio_events), 0)
                    a_ev = audio_events[0]
                    self.assertEqual(a_ev["format"], "mp3")
                    self.assertEqual(a_ev["audio"], b64_audio)
                    self.assertEqual(a_ev["sentenceIndex"], 0)
                    self.assertFalse(a_ev["isLast"])

    async def test_legacy_streaming_when_streaming_tts_disabled(self):
        settings.STREAMING_TTS_ENABLED = False

        payload = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Hi")],
            interactiveMode=True,
            character="meera"
        )
        fake_req = MagicMock(spec=Request)
        fake_req.is_disconnected = AsyncMock(return_value=False)

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", False):
            with patch.object(agent_service, "run", new=AsyncMock(return_value=AgentResult(
                content="Hello! How can I help you today?",
                model="mock-model",
                results=ChatResults()
            ))):
                response = await chat_stream_endpoint(payload, fake_req)
                events = []
                async for chunk in response.body_iterator:
                    for line in chunk.split("\n\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))

                types = [e["type"] for e in events]
                self.assertIn("turn_started", types)
                self.assertIn("chunk", types)
                self.assertNotIn("audio", types, "No audio events when STREAMING_TTS_ENABLED is False")

    async def test_tts_failure_still_shows_approved_text(self):
        """If streaming TTS connection fails, approved sentence is still delivered to client."""
        settings.STREAMING_TTS_ENABLED = True

        async def fake_agent_run(*args, **kwargs):
            if kwargs.get("on_sentence"):
                kwargs["on_sentence"]("Here are your travel options.")
            return AgentResult(
                content="Here are your travel options.",
                model="mock-model",
                results=ChatResults()
            )

        payload = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Options")],
            interactiveMode=True,
            character="meera"
        )
        fake_req = MagicMock(spec=Request)
        fake_req.is_disconnected = AsyncMock(return_value=False)

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", True):
            with patch.object(agent_service, "run", side_effect=fake_agent_run):
                # WebSocket connect fails
                with patch("websockets.connect", side_effect=ConnectionError("Sarvam service down")):
                    response = await chat_stream_endpoint(payload, fake_req)
                    events = []
                    async for chunk in response.body_iterator:
                        for line in chunk.split("\n\n"):
                            line = line.strip()
                            if line.startswith("data: "):
                                events.append(json.loads(line[6:]))

                    types = [e["type"] for e in events]
                    self.assertIn("sentence", types)
                    sentence_events = [e for e in events if e["type"] == "sentence"]
                    self.assertIn("Here are your travel options.", sentence_events[0]["delta"])
                    self.assertEqual(sentence_events[0].get("sentenceIndex"), 0)
                    # When streaming TTS connection fails, ttsMode falls back to 'rest'
                    turn_started_events = [e for e in events if e["type"] == "turn_started"]
                    self.assertTrue(len(turn_started_events) > 0)
                    self.assertEqual(turn_started_events[0].get("ttsMode"), "rest")
                    self.assertIn("status", types)
                    status_events = [e for e in events if e["type"] == "status"]
                    self.assertTrue(any("Voice audio is currently unavailable" in s.get("statusText", "") for s in status_events))

    async def test_tts_mode_advertised_and_sentence_indexing(self):
        """Verifies that turn_started advertises ttsMode='rest' when streaming TTS is disabled and sentences are indexed."""
        settings.STREAMING_TTS_ENABLED = False

        async def fake_agent_run(*args, **kwargs):
            if kwargs.get("on_sentence"):
                kwargs["on_sentence"]("Sentence 1.")
                kwargs["on_sentence"]("Sentence 2.")
            return AgentResult(
                content="Sentence 1. Sentence 2.",
                model="mock-model",
                results=ChatResults()
            )

        payload = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Test prompt")],
            interactiveMode=True,
            character="kabir"
        )
        fake_req = MagicMock(spec=Request)
        fake_req.is_disconnected = AsyncMock(return_value=False)

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", True):
            with patch.object(agent_service, "run", side_effect=fake_agent_run):
                response = await chat_stream_endpoint(payload, fake_req)
                events = []
                async for chunk in response.body_iterator:
                    for line in chunk.split("\n\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))

                turn_started = next(e for e in events if e["type"] == "turn_started")
                self.assertEqual(turn_started.get("ttsMode"), "rest")

                sentences = [e for e in events if e["type"] == "sentence"]
                self.assertEqual(len(sentences), 2)
                self.assertEqual(sentences[0]["sentenceIndex"], 0)
                self.assertEqual(sentences[1]["sentenceIndex"], 1)

    def test_bulbul_v4_flash_model_constraint(self):
        """Verifies that bulbul:v4-flash is the exclusive model and bulbul:v3 is never configured."""
        self.assertEqual(settings.SARVAM_TTS_MODEL, "bulbul:v4-flash")
        session = StreamingTtsSession(turn_id="t1", character="meera")
        self.assertEqual(session.model, "bulbul:v4-flash")
        self.assertNotIn("v3", session.model)


if __name__ == "__main__":
    unittest.main()
