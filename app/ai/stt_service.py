import asyncio
import base64
import collections
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Deque, List, Optional
import websockets

from ..config import settings

logger = logging.getLogger("stt_service")

# Terminal HTTP / WebSocket error codes that must not cause retry loops
TERMINAL_ERROR_CODES = {"401", "403", "429", "unauthorized", "forbidden", "quota_exceeded", "invalid_api_key"}


@dataclass
class SttEvent:
    kind: str  # "partial" | "final" | "speech_start" | "speech_end" | "status" | "error"
    text: str = ""
    language: str = "auto"
    engine: str = "unknown"
    error_code: Optional[str] = None
    is_fatal: bool = False


class BoundedAudioBuffer:
    """Bounded, prefix-preserving in-memory buffer of the current utterance (for failover replay).

    Never persists audio to disk. Default cap: 60s of 16kHz mono PCM16 (32,000 B/s) = 1,920,000 bytes.
    Keeps the BEGINNING of the utterance; once full, further frames are dropped and `overflowed`
    is set so callers never replay a tail and present it as the whole utterance.
    """

    def __init__(self, max_bytes: int = 1_920_000):
        self.max_bytes = max_bytes
        self._frames: Deque[bytes] = collections.deque()
        self._current_bytes: int = 0
        self.overflowed: bool = False

    def append(self, frame: bytes) -> bool:
        if not frame:
            return True
        if self.overflowed or self._current_bytes + len(frame) > self.max_bytes:
            self.overflowed = True
            return False
        self._frames.append(frame)
        self._current_bytes += len(frame)
        return True

    def get_all(self) -> List[bytes]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._current_bytes = 0
        self.overflowed = False


class SarvamRealtimeSttSession:
    """Manages a Sarvam AI Realtime STT WebSocket connection for one utterance."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ws_url: Optional[str] = None,
        stream_type: Optional[str] = None,
        endpointing: Optional[str] = None,
        silence_duration_ms: Optional[int] = None,
        connect_fn: Optional[Callable] = None,
    ):
        self.api_key = api_key if api_key is not None else getattr(settings, "SARVAM_API_KEY", "")
        self.model = model or getattr(settings, "SARVAM_STT_MODEL", "saaras:v3-realtime")
        self.ws_url = ws_url or getattr(settings, "SARVAM_STT_WS_URL", "wss://api.sarvam.ai/speech-to-text-realtime/ws")
        self.stream_type = stream_type or getattr(settings, "SARVAM_STT_STREAM_TYPE", "fast")
        self.endpointing = endpointing or getattr(settings, "SARVAM_STT_ENDPOINTING", "vad")
        self.silence_duration_ms = silence_duration_ms or getattr(settings, "SARVAM_STT_SILENCE_DURATION_MS", 2000)
        self._connect_fn = connect_fn or websockets.connect

        self.ws = None
        self.is_connected = False
        self.is_closed = False
        self._reader_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[Optional[SttEvent]] = asyncio.Queue()
        self._sample_rate: int = 16000
        self._language_code: str = "auto"

    def _build_url(self, sample_rate: int, language_code: str) -> str:
        # Normalize language to verified Sarvam BCP-47 codes: "auto", "hi-IN", "mr-IN", "en-IN"
        lang_map = {
            "auto": "auto",
            "hi": "hi-IN",
            "hi-in": "hi-IN",
            "mr": "mr-IN",
            "mr-in": "mr-IN",
            "en": "en-IN",
            "en-in": "en-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "bn": "bn-IN",
            "gu": "gu-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
            "pa": "pa-IN",
            "ur": "ur-IN",
        }
        lang = lang_map.get((language_code or "auto").lower(), language_code if language_code else "auto")
        # Validate sample rate: Sarvam requires 16000 or 8000
        rate = 16000 if sample_rate not in (8000, 16000) else sample_rate

        params = [
            f"model={self.model}",
            f"language_code={lang}",
            f"stream_type={self.stream_type}",
            "encoding=linear16",
            f"sample_rate={rate}",
            f"endpointing={self.endpointing}",
        ]
        if self.endpointing == "vad":
            params.append(f"silence_duration_ms={self.silence_duration_ms}")
            params.append("min_speech_duration_ms=250")
            params.append("threshold=0.3")

        return f"{self.ws_url}?{'&'.join(params)}"

    async def start(self, sample_rate: int = 16000, language_code: str = "auto") -> bool:
        if not self.api_key:
            logger.warning("SARVAM_API_KEY is not configured; cannot start Sarvam STT.")
            return False

        self._sample_rate = sample_rate
        self._language_code = language_code
        url = self._build_url(sample_rate, language_code)
        headers = {"api-subscription-key": self.api_key}

        try:
            self.ws = await self._connect_fn(url, additional_headers=headers)
            self.is_connected = True
            self._reader_task = asyncio.create_task(self._read_messages())
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Sarvam Realtime STT: {e}")
            self.is_connected = False
            return False

    async def send_audio(self, pcm_bytes: bytes) -> bool:
        if not self.is_connected or self.is_closed or not self.ws:
            return False
        try:
            # Sarvam allows up to 16,000 bytes per frame in 'fast' mode.
            # Slice incoming audio into <= 3,200 byte sub-chunks (100ms at 16kHz PCM16).
            chunk_size = 3200
            for offset in range(0, max(1, len(pcm_bytes)), chunk_size):
                sub = pcm_bytes[offset:offset + chunk_size]
                if not sub:
                    continue
                b64_audio = base64.b64encode(sub).decode("ascii")
                msg = json.dumps({"event": "audio_input", "audio": b64_audio})
                await self.ws.send(msg)
            return True
        except Exception as e:
            logger.warning(f"Error sending audio to Sarvam STT: {e}")
            return False

    async def _read_messages(self) -> None:
        received_final = False
        received_error = False
        last_partial = ""
        try:
            async for raw in self.ws:
                if self.is_closed:
                    break
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                event = data.get("event") or data.get("type", "")
                if event == "transcript.partial":
                    t = data.get("transcript") or data.get("text") or ""
                    last_partial = t
                    await self._event_queue.put(SttEvent(kind="partial", text=t, engine="sarvam"))
                elif event == "transcript.final":
                    received_final = True
                    t = data.get("transcript") or data.get("text") or ""
                    lang = data.get("language_code") or data.get("language") or self._language_code
                    await self._event_queue.put(SttEvent(kind="final", text=t, language=lang, engine="sarvam"))
                elif event in ("vad.speech_start", "speech_start"):
                    await self._event_queue.put(SttEvent(kind="speech_start", engine="sarvam"))
                elif event in ("vad.speech_end", "speech_end"):
                    await self._event_queue.put(SttEvent(kind="speech_end", engine="sarvam"))
                elif event in ("session.end", "end"):
                    break
                elif event == "error":
                    received_error = True
                    msg = data.get("message", "Sarvam STT error")
                    code = str(data.get("code", "stt_error"))
                    is_fatal = data.get("is_fatal", False) or code in TERMINAL_ERROR_CODES
                    logger.error(f"Sarvam STT provider error: {code} - {msg} (fatal={is_fatal})")
                    await self._event_queue.put(SttEvent(kind="error", text=msg, error_code=code, is_fatal=is_fatal, engine="sarvam"))
                    if is_fatal:
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            received_error = True
            logger.warning(f"Sarvam STT socket connection closed: {e}")
            await self._event_queue.put(SttEvent(kind="error", text=str(e), error_code="connection_loss", is_fatal=False, engine="sarvam"))
        finally:
            if not received_final and not received_error and not self.is_closed:
                if last_partial:
                    await self._event_queue.put(SttEvent(kind="final", text=last_partial, language=self._language_code, engine="sarvam"))
                else:
                    await self._event_queue.put(SttEvent(kind="error", text="Connection closed before final transcript", error_code="connection_loss", is_fatal=False, engine="sarvam"))
            await self._event_queue.put(None)

    async def receive_events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._event_queue.get()
            if ev is None or self.is_closed:
                break
            yield ev

    async def stop(self) -> None:
        """Gracefully ends audio streaming by notifying Sarvam with {"event": "end"}."""
        if self.is_closed or not self.ws:
            return
        try:
            await self.ws.send(json.dumps({"event": "end"}))
        except Exception as e:
            logger.warning(f"Error sending end event to Sarvam STT: {e}")

    async def cancel(self) -> None:
        """Immediately closes socket and releases reader."""
        self.is_closed = True
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        await self._event_queue.put(None)


import queue

class ParakeetSttSession:
    """Encapsulates NVIDIA Parakeet (Riva) streaming session."""

    def __init__(self, sample_rate: int = 16000, language_code: str = "auto", asr_service_fn: Optional[Callable] = None):
        self.sample_rate = sample_rate
        self.language_code = language_code
        self._asr_service_fn = asr_service_fn
        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._event_q: "asyncio.Queue[Optional[SttEvent]]" = asyncio.Queue()
        self.is_connected = False
        self.is_closed = False

    async def start(self, sample_rate: int = 16000, language_code: str = "auto") -> bool:
        from ..api.stt import _recog_config, _language_code, _join
        try:
            import riva.client
        except ImportError:
            logger.warning("nvidia-riva-client not installed; Riva STT streaming unavailable")
            self.is_connected = False
            return False

        self.sample_rate = sample_rate
        self.language_code = language_code
        self.is_connected = True
        loop = asyncio.get_running_loop()

        def worker(rate: int, lang: str):
            from ..api.stt import _asr_service
            finals = []
            interim = ""

            def chunks():
                while (c := self._audio_q.get()) is not None:
                    yield c

            try:
                service = self._asr_service_fn() if self._asr_service_fn else _asr_service()
                riva_lang = _language_code(lang)
                cfg = riva.client.StreamingRecognitionConfig(
                    config=_recog_config(rate, riva_lang),
                    interim_results=True,
                )
                for resp in service.streaming_response_generator(audio_chunks=chunks(), streaming_config=cfg):
                    for r in resp.results:
                        if not r.alternatives:
                            continue
                        t = r.alternatives[0].transcript.strip()
                        if r.is_final:
                            if t:
                                finals.append(t)
                            interim = ""
                        else:
                            interim = t
                        joined = _join(finals, interim)
                        loop.call_soon_threadsafe(self._event_q.put_nowait, SttEvent(kind="partial", text=joined, engine="parakeet"))
                final_text = _join(finals, interim)
                has_cyrillic = any("\u0400" <= ch <= "\u04FF" or "\u0500" <= ch <= "\u052F" for ch in final_text)
                final_lang = "ru" if has_cyrillic else riva_lang
                loop.call_soon_threadsafe(self._event_q.put_nowait, SttEvent(kind="final", text=final_text, language=final_lang, engine="parakeet"))
            except Exception as e:
                logger.warning(f"Parakeet ASR worker error: {e}")
                loop.call_soon_threadsafe(self._event_q.put_nowait, SttEvent(kind="error", text=str(e), error_code="parakeet_failed", engine="parakeet"))
            finally:
                loop.call_soon_threadsafe(self._event_q.put_nowait, None)

        loop.run_in_executor(None, worker, self.sample_rate, self.language_code)
        return True

    async def send_audio(self, pcm_bytes: bytes) -> bool:
        if not self.is_connected or self.is_closed or getattr(self, "_stopping", False):
            return False
        self._audio_q.put(pcm_bytes)
        return True

    async def receive_events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._event_q.get()
            if ev is None or self.is_closed:
                break
            yield ev

    async def stop(self) -> None:
        """Stops accepting further audio and signals worker to finish and emit final SttEvent."""
        self._stopping = True
        logger.info("Parakeet session stop requested; queued EOF to worker")
        self._audio_q.put(None)

    async def cancel(self) -> None:
        """Immediately closes session, discards queued audio, and interrupts worker."""
        self.is_closed = True
        self._stopping = True
        logger.info("Parakeet session cancelled")
        self._audio_q.put(None)


class FailoverSttSession:
    """Unified STT session providing transparent failover from Sarvam Realtime STT to Parakeet.

    Maintains a bounded in-memory buffer (5s / 160KB). If Sarvam fails on connect or during speech,
    it seamlessly falls back to Parakeet by replaying buffered audio.
    """

    def __init__(
        self,
        sarvam_enabled: Optional[bool] = None,
        sarvam_session: Optional[SarvamRealtimeSttSession] = None,
        parakeet_session: Optional[ParakeetSttSession] = None,
        dual_mode: bool = False,
    ):
        self.sarvam_enabled = sarvam_enabled if sarvam_enabled is not None else getattr(settings, "SARVAM_STT_ENABLED", False)
        self.sarvam: Optional[SarvamRealtimeSttSession] = sarvam_session
        self.parakeet: Optional[ParakeetSttSession] = parakeet_session
        self._allow_dual_mode = dual_mode or (getattr(settings, "STT_DUAL_MODE_ENABLED", False) and parakeet_session is None)
        self.active_provider: str = "none"
        self._audio_buffer = BoundedAudioBuffer()
        self._event_queue: asyncio.Queue[Optional[SttEvent]] = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._is_closed = False
        self._stopping = False
        self._sample_rate: int = 16000
        self._language_code: str = "auto"
        self._final_emitted = False
        self._frames_sent = 0
        self._bytes_sent = 0
        self._dual_mode = False

    async def start(self, sample_rate: int = 16000, language_code: str = "auto") -> bool:
        self._sample_rate = sample_rate
        self._language_code = language_code

        # If Sarvam is enabled, attempt Sarvam first
        if self.sarvam_enabled:
            if not self.sarvam:
                self.sarvam = SarvamRealtimeSttSession()
            connected = await self.sarvam.start(sample_rate, language_code)
            if connected:
                self.active_provider = "sarvam"
                # Start Parakeet in parallel only if dual-mode is permitted
                if self._allow_dual_mode:
                    if not self.parakeet:
                        self.parakeet = ParakeetSttSession(sample_rate, language_code)
                    try:
                        p_ok = await self.parakeet.start(sample_rate, language_code)
                        if p_ok:
                            self._dual_mode = True
                            logger.info("Parallel Parakeet active for clean real-time text viewing.")
                    except Exception as e:
                        logger.debug(f"Parallel Parakeet unavailable: {e}")
                self._pump_task = asyncio.create_task(self._pump_sarvam_events())
                return True
            else:
                logger.info("Sarvam connection failed at startup; failing over to Parakeet.")
                await self._event_queue.put(SttEvent(kind="status", text="Sarvam unavailable, using backup speech recognizer.", engine="parakeet"))

        # Fallback to Parakeet
        return await self._start_parakeet(sample_rate, language_code)

    async def _start_parakeet(self, sample_rate: int, language_code: str) -> bool:
        self.active_provider = "parakeet"
        if not self.parakeet:
            self.parakeet = ParakeetSttSession(sample_rate, language_code)
        connected = await self.parakeet.start(sample_rate, language_code)
        if connected:
            self._pump_task = asyncio.create_task(self._pump_parakeet_events())
            return True
        else:
            await self._event_queue.put(SttEvent(kind="error", text="All STT providers unavailable", error_code="stt_unavailable", is_fatal=True))
            await self._event_queue.put(None)
            return False

    async def _pump_sarvam_events(self) -> None:
        sarvam = self.sarvam
        assert sarvam is not None
        failed_over = False

        # Read Parakeet partials concurrently for crisp real-time text viewing on Flutter
        async def _read_parakeet_partials():
            if not self.parakeet:
                return
            try:
                async for p_ev in self.parakeet.receive_events():
                    if self.active_provider != "sarvam":
                        break
                    if p_ev.kind == "partial" and not self._final_emitted:
                        await self._event_queue.put(p_ev)
            except Exception:
                pass

        parakeet_partial_task = asyncio.create_task(_read_parakeet_partials())
        try:
            async for ev in sarvam.receive_events():
                if self.active_provider != "sarvam":
                    break  # Discard late events after failover
                if ev.kind == "error":
                    if ev.is_fatal:
                        # Terminal error (quota, auth) - report error and switch to Parakeet without infinite retrying
                        logger.warning(f"Sarvam terminal error ({ev.error_code}): {ev.text}. Failing over to Parakeet.")
                        failed_over = True
                        break
                    else:
                        # Non-terminal recoverable error (e.g. connection loss)
                        logger.warning(f"Sarvam recoverable error ({ev.error_code}): {ev.text}. Failing over to Parakeet.")
                        failed_over = True
                        break
                elif ev.kind == "final":
                    if not self._final_emitted:
                        self._final_emitted = True
                        self._audio_buffer.clear()
                        await self._event_queue.put(ev)
                elif ev.kind == "partial":
                    # If Parakeet is running in dual mode, Parakeet provides crisp Latin partials; otherwise use Sarvam partials
                    if not self._dual_mode:
                        await self._event_queue.put(ev)
                else:
                    await self._event_queue.put(ev)
        except Exception as e:
            logger.warning(f"Exception during Sarvam event pump: {e}")
            failed_over = True
        finally:
            parakeet_partial_task.cancel()

        if failed_over and not self._is_closed and not self._final_emitted:
            await self._failover_to_parakeet()
        elif not failed_over:
            await self._event_queue.put(None)

    async def _failover_to_parakeet(self) -> None:
        self._dual_mode = False
        logger.info("Executing failover to Parakeet with bounded audio buffer replay...")
        # 1. Cancel / close Sarvam session
        if self.sarvam:
            await self.sarvam.cancel()

        # 2. Inform client via status event
        await self._event_queue.put(SttEvent(kind="status", text="Switching to backup speech recognizer", engine="parakeet"))

        # 3. Start Parakeet session
        ok = await self._start_parakeet(self._sample_rate, self._language_code)
        if not ok:
            return

        # 4. Replay the full buffered utterance into Parakeet (never a tail-only replay)
        if self._audio_buffer.overflowed:
            await self._event_queue.put(SttEvent(
                kind="error", text="That was too long to recover. Please repeat the last part.",
                error_code="utterance_too_long", is_fatal=True, engine="parakeet"))
            return
        frames = self._audio_buffer.get_all()
        if self.parakeet and frames:
            logger.info(f"Replaying {len(frames)} buffered audio frames to Parakeet...")
            for f in frames:
                await self.parakeet.send_audio(f)
        if self._stopping and self.parakeet:
            await self.parakeet.stop()  # stop() arrived before failover; finalize the replacement

    async def _pump_parakeet_events(self) -> None:
        assert self.parakeet is not None
        try:
            async for ev in self.parakeet.receive_events():
                if self.active_provider == "parakeet":
                    if ev.kind == "final":
                        self._final_emitted = True
                    await self._event_queue.put(ev)
        except Exception as e:
            logger.warning(f"Exception during Parakeet event pump: {e}")
            await self._event_queue.put(SttEvent(kind="error", text=str(e), error_code="parakeet_error", is_fatal=True, engine="parakeet"))
        finally:
            await self._event_queue.put(None)

    async def send_audio(self, pcm_bytes: bytes) -> bool:
        if self._is_closed or self._stopping:
            return False
        self._frames_sent += 1
        self._bytes_sent += len(pcm_bytes)
        if self._frames_sent % 50 == 0:
            logger.debug(f"FailoverSttSession streaming: provider={self.active_provider}, frames={self._frames_sent}, bytes={self._bytes_sent}")
        # Save to circular bounded buffer for possible failover
        self._audio_buffer.append(pcm_bytes)

        if self.active_provider == "sarvam" and self.sarvam:
            res = await self.sarvam.send_audio(pcm_bytes)
            if self._dual_mode and self.parakeet and not getattr(self.parakeet, "is_closed", False):
                await self.parakeet.send_audio(pcm_bytes)
            return res
        elif self.active_provider == "parakeet" and self.parakeet:
            return await self.parakeet.send_audio(pcm_bytes)
        return False

    async def receive_events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._event_queue.get()
            if ev is None or self._is_closed:
                break
            yield ev

    async def stop(self) -> None:
        """Stops accepting further audio, instructs active provider to finalize,
        and allows the final transcript to be emitted."""
        self._stopping = True  # buffer kept until final/cancel so failover can still replay the whole utterance
        logger.info(f"FailoverSttSession stop: active_provider={self.active_provider}, total_frames={self._frames_sent}, total_bytes={self._bytes_sent}")
        if self.active_provider == "sarvam" and self.sarvam:
            await self.sarvam.stop()
            if self.parakeet and not getattr(self.parakeet, "is_closed", False):
                await self.parakeet.stop()
        elif self.active_provider == "parakeet" and self.parakeet:
            await self.parakeet.stop()

    async def cancel(self) -> None:
        """Immediately tears down sessions, cancels tasks, and releases resources."""
        self._is_closed = True
        self._stopping = True
        logger.info(f"FailoverSttSession cancel: active_provider={self.active_provider}, total_frames={self._frames_sent}, total_bytes={self._bytes_sent}")
        if self.sarvam:
            await self.sarvam.cancel()
        if self.parakeet:
            await self.parakeet.cancel()
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        self._audio_buffer.clear()
        await self._event_queue.put(None)
