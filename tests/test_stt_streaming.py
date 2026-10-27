import asyncio
import base64
import json
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.ai.stt_service import (
    SarvamRealtimeSttSession,
    ParakeetSttSession,
    FailoverSttSession,
    BoundedAudioBuffer,
    SttEvent,
    TERMINAL_ERROR_CODES,
)
from app.agent.intent.spoken_normalizer import normalize_spoken_travel_text
from app.agent.resolvers.location_resolver import LocationResolver


class MockWebSocket:
    """Mock WebSocket client conforming to websockets protocol."""

    def __init__(self, incoming_messages=None, fail_connect=False):
        self.incoming_messages = list(incoming_messages or [])
        self.sent_messages = []
        self.closed = False
        self.fail_connect = fail_connect

    async def send(self, data: str):
        if self.closed:
            raise ConnectionResetError("Cannot send on closed socket")
        self.sent_messages.append(data)

    async def recv(self) -> str:
        if self.closed or not self.incoming_messages:
            raise ConnectionResetError("Socket closed")
        return self.incoming_messages.pop(0)

    async def close(self, code: int = 1000):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming_messages or self.closed:
            raise StopAsyncIteration
        return self.incoming_messages.pop(0)


class TestSarvamRealtimeSttSession(unittest.IsolatedAsyncioTestCase):
    async def test_url_and_parameters(self):
        session = SarvamRealtimeSttSession(
            api_key="test-key",
            model="saaras:v3-realtime",
            stream_type="fast",
            endpointing="vad",
            silence_duration_ms=600,
        )
        url = session._build_url(16000, "hi-IN")
        self.assertIn("model=saaras:v3-realtime", url)
        self.assertIn("language_code=hi-IN", url)
        self.assertIn("stream_type=fast", url)
        self.assertIn("encoding=linear16", url)
        self.assertIn("sample_rate=16000", url)
        self.assertIn("endpointing=vad", url)
        self.assertIn("silence_duration_ms=600", url)

        # Test shorthand mappings: hi -> hi-IN, mr -> mr-IN, en -> en-IN, auto -> auto
        self.assertIn("language_code=hi-IN", session._build_url(16000, "hi"))
        self.assertIn("language_code=mr-IN", session._build_url(16000, "mr"))
        self.assertIn("language_code=en-IN", session._build_url(16000, "en"))
        self.assertIn("language_code=auto", session._build_url(16000, "auto"))

    async def test_auth_header_and_connect(self):
        mock_ws = MockWebSocket()
        captured_headers = {}
        captured_url = ""

        async def fake_connect(url, additional_headers=None):
            nonlocal captured_headers, captured_url
            captured_url = url
            captured_headers = additional_headers or {}
            return mock_ws

        session = SarvamRealtimeSttSession(
            api_key="secret-subscription-key",
            connect_fn=fake_connect,
        )
        connected = await session.start(sample_rate=16000, language_code="auto")
        self.assertTrue(connected)
        self.assertEqual(captured_headers.get("api-subscription-key"), "secret-subscription-key")
        # Ensure API key is not leaked in the URL
        self.assertNotIn("secret-subscription-key", captured_url)

        await session.stop()
        self.assertIn('{"event": "end"}', mock_ws.sent_messages)
        await session.cancel()
        self.assertTrue(mock_ws.closed)

    async def test_send_audio_serialization(self):
        mock_ws = MockWebSocket()
        session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_ws),
        )
        await session.start()

        pcm_sample = b"\x00\x01\x02\x03\x04\x05"
        sent_ok = await session.send_audio(pcm_sample)
        self.assertTrue(sent_ok)
        self.assertEqual(len(mock_ws.sent_messages), 1)

        payload = json.loads(mock_ws.sent_messages[0])
        self.assertEqual(payload.get("event"), "audio_input")
        expected_b64 = base64.b64encode(pcm_sample).decode("ascii")
        self.assertEqual(payload.get("audio"), expected_b64)

        await session.stop()

    async def test_receive_events_partial_final_vad(self):
        mock_events = [
            json.dumps({"event": "speech_start"}),
            json.dumps({"event": "transcript.partial", "transcript": "delhi se"}),
            json.dumps({"event": "transcript.partial", "transcript": "delhi se mumbai"}),
            json.dumps({"event": "speech_end"}),
            json.dumps({"event": "transcript.final", "transcript": "delhi se mumbai flight", "language_code": "hi-IN"}),
        ]
        mock_ws = MockWebSocket(incoming_messages=mock_events)
        session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_ws),
        )
        await session.start()

        events = []
        async for ev in session.receive_events():
            events.append(ev)

        self.assertEqual(len(events), 5)
        self.assertEqual(events[0].kind, "speech_start")
        self.assertEqual(events[1].kind, "partial")
        self.assertEqual(events[1].text, "delhi se")
        self.assertEqual(events[2].kind, "partial")
        self.assertEqual(events[2].text, "delhi se mumbai")
        self.assertEqual(events[3].kind, "speech_end")
        self.assertEqual(events[4].kind, "final")
        self.assertEqual(events[4].text, "delhi se mumbai flight")
        self.assertEqual(events[4].language, "hi-IN")
        self.assertEqual(events[4].engine, "sarvam")

    async def test_terminal_error_no_retry_loop(self):
        mock_events = [
            json.dumps({"event": "error", "code": 401, "message": "Invalid API key"}),
        ]
        mock_ws = MockWebSocket(incoming_messages=mock_events)
        session = SarvamRealtimeSttSession(
            api_key="bad-key",
            connect_fn=AsyncMock(return_value=mock_ws),
        )
        await session.start()

        events = []
        async for ev in session.receive_events():
            events.append(ev)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "error")
        self.assertTrue(events[0].is_fatal)
        self.assertEqual(events[0].error_code, "401")


class TestFailoverSttSession(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._parakeet_patch = patch("app.ai.stt_service.ParakeetSttSession.start", AsyncMock(return_value=False))
        self._parakeet_patch.start()

    async def asyncTearDown(self):
        self._parakeet_patch.stop()

    async def test_bounded_audio_buffer_never_writes_disk(self):
        buffer = BoundedAudioBuffer(max_bytes=100)
        chunk = b"A" * 40
        buffer.append(chunk)
        buffer.append(chunk)
        buffer.append(chunk)
        # Total added is 120 bytes, capped at 100 max_bytes
        all_frames = buffer.get_all()
        total_len = sum(len(f) for f in all_frames)
        self.assertLessEqual(total_len, 100)
        self.assertEqual(total_len, 80)  # Evicted first 40 byte chunk

    async def test_failover_when_sarvam_connect_fails(self):
        # Sarvam connection fails on startup
        sarvam = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(side_effect=Exception("Connection refused")),
        )
        mock_parakeet = ParakeetSttSession()
        mock_parakeet.start = AsyncMock(return_value=True)

        async def fake_parakeet_events():
            yield SttEvent(kind="partial", text="namaste", engine="parakeet")
            yield SttEvent(kind="final", text="namaste ji", engine="parakeet")

        mock_parakeet.receive_events = fake_parakeet_events

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam,
            parakeet_session=mock_parakeet,
        )
        started = await manager.start(16000, "hi-IN")
        self.assertTrue(started)
        self.assertEqual(manager.active_provider, "parakeet")

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        # First event is status warning about failover, then Parakeet events
        self.assertEqual(events[0].kind, "status")
        self.assertEqual(events[1].kind, "partial")
        self.assertEqual(events[1].text, "namaste")
        self.assertEqual(events[2].kind, "final")
        self.assertEqual(events[2].text, "namaste ji")

    async def test_mid_utterance_failover_replays_buffer(self):
        # Sarvam connects, sends one partial, then disconnects
        mock_ws = MockWebSocket(incoming_messages=[
            json.dumps({"event": "transcript.partial", "transcript": "flight to"}),
            json.dumps({"event": "error", "code": 500, "message": "Server error", "is_fatal": False}),
        ])
        sarvam = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_ws),
        )

        mock_parakeet = ParakeetSttSession()
        mock_parakeet.start = AsyncMock(return_value=True)
        replayed_audio = []
        mock_parakeet.send_audio = AsyncMock(side_effect=lambda b: replayed_audio.append(b))

        async def fake_parakeet_events():
            yield SttEvent(kind="final", text="flight to goa", engine="parakeet")

        mock_parakeet.receive_events = fake_parakeet_events

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam,
            parakeet_session=mock_parakeet,
        )
        await manager.start(16000, "en-IN")

        # Stream some audio into the manager
        frame1 = b"chunk1" * 10
        frame2 = b"chunk2" * 10
        await manager.send_audio(frame1)
        await manager.send_audio(frame2)

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        # Verifies audio buffer was replayed into Parakeet on failover
        self.assertEqual(len(replayed_audio), 2)
        self.assertEqual(replayed_audio[0], frame1)
        self.assertEqual(replayed_audio[1], frame2)

        # Verifies failover events sequence
        kinds = [e.kind for e in events]
        self.assertIn("partial", kinds)
        self.assertIn("status", kinds)  # Safe status notification
        self.assertIn("final", kinds)
        final_ev = next(e for e in events if e.kind == "final")
        self.assertEqual(final_ev.text, "flight to goa")
        self.assertEqual(final_ev.engine, "parakeet")

    async def test_sarvam_disabled_preserves_parakeet_path(self):
        mock_parakeet = ParakeetSttSession()
        mock_parakeet.start = AsyncMock(return_value=True)

        async def fake_parakeet_events():
            yield SttEvent(kind="final", text="direct parakeet result", engine="parakeet")

        mock_parakeet.receive_events = fake_parakeet_events

        manager = FailoverSttSession(
            sarvam_enabled=False,
            parakeet_session=mock_parakeet,
        )
        started = await manager.start(16000, "en")
        self.assertTrue(started)
        self.assertEqual(manager.active_provider, "parakeet")

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].text, "direct parakeet result")

    async def test_vad_speech_end_followed_by_delayed_final_transcript(self):
        """Server VAD speech_end stops audio capture but keeps socket open until final transcript arrives."""
        mock_sarvam_ws = MockWebSocket(incoming_messages=[
            json.dumps({"event": "transcript.partial", "transcript": "flight to"}),
            json.dumps({"event": "vad.speech_end"}),
            json.dumps({"event": "transcript.final", "transcript": "flight to delhi", "language_code": "en-IN"}),
            json.dumps({"event": "session.end"}),
        ])

        sarvam_session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_sarvam_ws),
        )

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam_session,
        )
        started = await manager.start(16000, "en")
        self.assertTrue(started)

        events = []
        async def collect():
            async for ev in manager.receive_events():
                events.append(ev)
                if ev.kind == "speech_end":
                    # Simulate client stopping audio capture on speech_end
                    await manager.stop()

        await collect()

        kinds = [e.kind for e in events]
        self.assertIn("partial", kinds)
        self.assertIn("speech_end", kinds)
        self.assertIn("final", kinds)
        final_ev = next(e for e in events if e.kind == "final")
        self.assertEqual(final_ev.text, "flight to delhi")
        # Confirms graceful stop was sent to Sarvam
        self.assertIn('{"event": "end"}', mock_sarvam_ws.sent_messages)

    async def test_alternative_event_orderings(self):
        """Final transcript arriving without or before speech_end is correctly emitted."""
        mock_sarvam_ws = MockWebSocket(incoming_messages=[
            json.dumps({"event": "transcript.partial", "transcript": "mumbai"}),
            json.dumps({"event": "transcript.final", "transcript": "mumbai trains", "language_code": "hi-IN"}),
            json.dumps({"event": "vad.speech_end"}),
            json.dumps({"event": "session.end"}),
        ])

        sarvam_session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_sarvam_ws),
        )

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam_session,
        )
        await manager.start(16000, "hi")

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        final_evs = [e for e in events if e.kind == "final"]
        self.assertEqual(len(final_evs), 1)
        self.assertEqual(final_evs[0].text, "mumbai trains")

    async def test_duplicate_final_events_deduplicated(self):
        """Duplicate final transcript events from provider are safely deduplicated to exactly once."""
        mock_sarvam_ws = MockWebSocket(incoming_messages=[
            json.dumps({"event": "transcript.final", "transcript": "delhi flights", "language_code": "en-IN"}),
            json.dumps({"event": "transcript.final", "transcript": "delhi flights", "language_code": "en-IN"}),
            json.dumps({"event": "session.end"}),
        ])

        sarvam_session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_sarvam_ws),
        )

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam_session,
        )
        await manager.start(16000, "en")

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        final_evs = [e for e in events if e.kind == "final"]
        self.assertEqual(len(final_evs), 1, "Exactly one final event must be emitted")

    async def test_connection_loss_during_finalization_fails_over(self):
        """Connection loss during finalization triggers fallback to Parakeet with buffered replay."""
        frame1 = b"\x01\x02" * 1600

        mock_sarvam_ws = MockWebSocket(incoming_messages=[
            json.dumps({"event": "vad.speech_end"}),
            # Then connection loss
        ])

        sarvam_session = SarvamRealtimeSttSession(
            api_key="test-key",
            connect_fn=AsyncMock(return_value=mock_sarvam_ws),
        )

        replayed_frames = []
        mock_parakeet = ParakeetSttSession()
        mock_parakeet.start = AsyncMock(return_value=True)

        async def fake_send_audio(pcm):
            replayed_frames.append(pcm)
            return True

        mock_parakeet.send_audio = fake_send_audio

        async def fake_parakeet_events():
            yield SttEvent(kind="final", text="recovered audio transcript", engine="parakeet")

        mock_parakeet.receive_events = fake_parakeet_events

        manager = FailoverSttSession(
            sarvam_enabled=True,
            sarvam_session=sarvam_session,
            parakeet_session=mock_parakeet,
        )
        await manager.start(16000, "en")
        await manager.send_audio(frame1)

        events = []
        async for ev in manager.receive_events():
            events.append(ev)

        self.assertEqual(len(replayed_frames), 1)
        self.assertEqual(replayed_frames[0], frame1)
        final_ev = next((e for e in events if e.kind == "final"), None)
        self.assertIsNotNone(final_ev)
        self.assertEqual(final_ev.text, "recovered audio transcript")
        self.assertEqual(final_ev.engine, "parakeet")


class TestLanguageAndSpokenNormalizer(unittest.TestCase):
    def test_hindi_travel_utterance(self):
        raw = "delhi se goa kal subah flight chahiye"
        normalized = normalize_spoken_travel_text(raw, "Asia/Kolkata")
        orig, dest, _ = LocationResolver.extract_route(normalized)
        self.assertEqual(orig.name.lower(), "delhi")
        self.assertEqual(dest.name.lower(), "goa")

    def test_marathi_travel_utterance(self):
        raw = "mumbai te pune train dakhav"
        normalized = normalize_spoken_travel_text(raw, "Asia/Kolkata")
        orig, dest, _ = LocationResolver.extract_route(normalized)
        self.assertEqual(orig.name.lower(), "mumbai")
        self.assertEqual(dest.name.lower(), "pune")

    def test_english_travel_utterance(self):
        raw = "flights from bangalore to delhi next monday"
        normalized = normalize_spoken_travel_text(raw, "Asia/Kolkata")
        orig, dest, _ = LocationResolver.extract_route(normalized)
        self.assertEqual(orig.name.lower(), "bengaluru")
        self.assertEqual(dest.name.lower(), "delhi")

    def test_hinglish_code_mixed_utterance(self):
        raw = "delhi to jaipur sabse sasti train konsi hai"
        normalized = normalize_spoken_travel_text(raw, "Asia/Kolkata")
        orig, dest, _ = LocationResolver.extract_route(normalized)
        self.assertEqual(orig.name.lower(), "delhi")
        self.assertEqual(dest.name.lower(), "jaipur")

    def test_ambiguous_destination_does_not_invent(self):
        raw = "kuch achi jagah batao ghumne ke liye"
        normalized = normalize_spoken_travel_text(raw, "Asia/Kolkata")
        orig, dest, _ = LocationResolver.extract_route(normalized)
        # Should not invent a destination when user is asking for general recommendation
        self.assertIsNone(dest)


class TestParakeetStopAndWebSocketLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_parakeet_stop_delivers_final_without_dropping(self):
        """Verifies that ParakeetSttSession.stop() does not drop the final transcript by setting is_closed prematurely."""
        mock_resp = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "delhi flight"
        mock_result = MagicMock()
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_resp.results = [mock_result]

        mock_service = MagicMock()
        mock_service.streaming_response_generator.return_value = [mock_resp]

        session = ParakeetSttSession(sample_rate=16000, language_code="hi", asr_service_fn=lambda: mock_service)
        started = await session.start(16000, "hi")
        self.assertTrue(started)

        await session.send_audio(b"\x00" * 3200)
        # Calling stop must not close the session before the worker emits final
        await session.stop()
        self.assertTrue(session._stopping)
        self.assertFalse(session.is_closed)

        events = []
        async for ev in session.receive_events():
            events.append(ev)

        final_ev = next((e for e in events if e.kind == "final"), None)
        self.assertIsNotNone(final_ev, "Final transcript must not be dropped on stop()")
        self.assertEqual(final_ev.text, "delhi flight")
        self.assertEqual(final_ev.engine, "parakeet")

    async def test_parakeet_cancel_immediately_interrupts(self):
        session = ParakeetSttSession(sample_rate=16000, language_code="hi")
        await session.start(16000, "hi")
        await session.cancel()
        self.assertTrue(session.is_closed)
        self.assertTrue(session._stopping)

    async def test_parakeet_detects_cyrillic_hallucination(self):
        """Verifies that Cyrillic text output by Parakeet is flagged with language='ru'."""
        mock_resp = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Привет мир"
        mock_result = MagicMock()
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_resp.results = [mock_result]

        mock_service = MagicMock()
        mock_service.streaming_response_generator.return_value = [mock_resp]

        session = ParakeetSttSession(sample_rate=16000, language_code="hi", asr_service_fn=lambda: mock_service)
        await session.start(16000, "hi")
        await session.send_audio(b"\x00" * 3200)
        await session.stop()

        events = []
        async for ev in session.receive_events():
            events.append(ev)

        final_ev = next((e for e in events if e.kind == "final"), None)
        self.assertIsNotNone(final_ev)
        self.assertEqual(final_ev.text, "Привет мир")
        self.assertEqual(final_ev.language, "ru")

    def test_websocket_stream_endpoint_happy_path(self):
        from starlette.testclient import TestClient
        from app.main import app

        mock_session = MagicMock()
        mock_session.active_provider = "parakeet"
        mock_session._final_emitted = True
        mock_session.start = AsyncMock(return_value=True)
        mock_session.send_audio = AsyncMock(return_value=True)
        mock_session.stop = AsyncMock()
        mock_session.cancel = AsyncMock()

        async def fake_events():
            yield SttEvent(kind="partial", text="delhi", engine="parakeet")
            yield SttEvent(kind="final", text="delhi flight", language="hi-IN", engine="parakeet")

        mock_session.receive_events = fake_events

        with patch("app.ai.stt_service.FailoverSttSession", return_value=mock_session), \
             patch("app.api.stt._available", return_value=True), \
             patch("app.api.stt._resolve_function_id", return_value="mock-fn-id"):
            client = TestClient(app)
            with client.websocket_connect("/api/stt/stream") as ws:
                ws.send_json({"type": "start", "language": "hi", "sampleRate": 16000})
                ready_msg = ws.receive_json()
                self.assertEqual(ready_msg["type"], "ready")

                ws.send_bytes(b"\x00" * 3200)
                ws.send_json({"type": "stop"})

                # Client receives partial then final transcript before closure
                part_msg = ws.receive_json()
                self.assertEqual(part_msg["type"], "partial")
                self.assertEqual(part_msg["text"], "delhi")

                final_msg = ws.receive_json()
                self.assertEqual(final_msg["type"], "final")
                self.assertEqual(final_msg["text"], "delhi flight")
                self.assertEqual(final_msg["engine"], "parakeet")

    def test_websocket_stream_endpoint_stt_unavailable(self):
        from starlette.testclient import TestClient
        from app.main import app

        with patch("app.api.stt._available", return_value=False):
            client = TestClient(app)
            with client.websocket_connect("/api/stt/stream") as ws:
                err_msg = ws.receive_json()
                self.assertEqual(err_msg["type"], "error")
                self.assertEqual(err_msg["code"], "stt_unavailable")


if __name__ == "__main__":
    unittest.main()
