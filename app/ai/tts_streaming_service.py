import re
import json
import logging
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
import websockets

from ..config import settings

logger = logging.getLogger("tts_streaming_service")

# Transliterate common Devanagari script slips into Roman Hinglish
_DEVANAGARI_MAP = {
    "सुबह": "subah", "शाम": "shaam", "रात": "raat", "दोपहर": "dopahar",
    "बजे": "baje", "रुपये": "₹", "रूपये": "₹", "हवाई": "hawai", "अड्डा": "adda",
    "टिकट": "ticket", "फ्लाइट": "flight", "ट्रेन": "train", "सस्ता": "sasta", "सस्ती": "sasti",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    "।": ".", "॥": "."
}


def sanitize_for_tts(text: str) -> str:
    """Sanitizes conversational output into clean Romanized Hinglish/English for TTS.

    - Replaces Devanagari slips and numbers with Romanized words/digits.
    - Strips markdown formatting (e.g. **bold**, *italics*, `code`).
    - Strips emojis and unusual symbols.
    - Preserves standard currency (₹, Rs), punctuation, and times.
    """
    if not text or not isinstance(text, str):
        return ""

    s = text

    # Strip markdown formatting
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Transliterate known Devanagari words/digits
    for dev, rom in _DEVANAGARI_MAP.items():
        s = s.replace(dev, rom)

    # Strip remaining Devanagari characters if any slipped through
    s = re.sub(r"[\u0900-\u097f]+", "", s)

    # Strip emojis and pictographs
    s = re.sub(
        r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe00-\ufe0f]",
        "",
        s,
    )

    # Clean multiple spaces and normalize punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class TtsAudioChunk:
    sentence_index: int
    chunk_index: int
    format: str
    audio: str  # Base64 string
    is_last: bool = False


class StreamingTtsSession:
    """Manages a Sarvam Bulbul WebSocket streaming TTS session for one turn.

    Features:
    - Opens a WebSocket connection to Sarvam TTS streaming API.
    - Sends connection configuration frame before any text.
    - Exclusively uses bulbul:v4-flash; never falls back to bulbul:v3.
    - Accepts approved, sanitized sentences in order.
    - Streams returned audio chunks asynchronously.
    - Flushes remaining text and closes the socket cleanly on turn completion or cancellation.
    - Never leaks API keys in logs or outputs.
    """

    def __init__(
        self,
        turn_id: str,
        character: str = "meera",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ws_url: Optional[str] = None,
        connect_fn: Optional[Callable] = None,
    ):
        self.turn_id: str = turn_id
        self.character: str = character.lower()
        self.model: str = model or getattr(settings, "SARVAM_TTS_MODEL", "bulbul:v4-flash")
        self.api_key: str = api_key if api_key is not None else getattr(settings, "SARVAM_API_KEY", "")
        self.ws_url: str = ws_url or getattr(settings, "SARVAM_TTS_WS_URL", "wss://api.sarvam.ai/text-to-speech/ws")
        self._connect_fn = connect_fn or websockets.connect

        self.ws = None
        self.is_connected: bool = False
        self.is_closed: bool = False
        self._reader_task: Optional[asyncio.Task] = None
        self._audio_queue: asyncio.Queue[Optional[TtsAudioChunk]] = asyncio.Queue()
        self._current_sentence_index: int = 0
        self._chunk_index_counter: int = 0

    def _get_speaker(self) -> str:
        return "kabir_hi_conversational" if self.character == "kabir" else "priya_hi_recovery"

    async def connect(self) -> bool:
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not configured; streaming TTS disabled for this turn.")
            return False

        headers = {"api-subscription-key": self.api_key}
        url = f"{self.ws_url}?model={self.model}&send_completion_event=true"

        try:
            self.ws = await self._connect_fn(url, additional_headers=headers)
            self.is_connected = True
            await self._send_config()
            self._reader_task = asyncio.create_task(self._read_messages())
            return True
        except Exception as e:
            logger.warning(
                f"Sarvam TTS connection with {self.model} failed: {e}. Audio disabled, text reply preserved."
            )
            self.is_connected = False
            return False

    async def _send_config(self):
        speaker = self._get_speaker()
        config_payload = {
            "type": "config",
            "data": {
                "model": self.model,
                "language_code": "hi-IN",
                "speaker": speaker,
                "pace": 1.0,
                "temperature": 0.6,
                "speech_sample_rate": 22050,
                "output_audio_codec": "mp3",
                "output_audio_bitrate": "128k",
                "min_buffer_size": 30,
                "max_chunk_length": 150,
            },
        }
        await self.ws.send(json.dumps(config_payload))

    async def send_sentence(self, sentence: str, sentence_index: int) -> bool:
        if not self.is_connected or self.is_closed or not self.ws:
            return False

        clean_text = sanitize_for_tts(sentence)
        if not clean_text:
            return False

        self._current_sentence_index = sentence_index
        text_payload = {
            "type": "text",
            "data": {
                "text": clean_text
            }
        }
        try:
            await self.ws.send(json.dumps(text_payload))
            return True
        except Exception as e:
            logger.error(f"Failed to send sentence to Sarvam TTS: {e}")
            return False

    async def flush(self) -> bool:
        if not self.is_connected or self.is_closed or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({"type": "flush"}))
            return True
        except Exception as e:
            logger.error(f"Failed to send flush to Sarvam TTS: {e}")
            return False

    async def _read_messages(self):
        try:
            async for raw_msg in self.ws:
                if self.is_closed:
                    break
                try:
                    msg = json.loads(raw_msg)
                except Exception:
                    continue

                msg_type = msg.get("type")
                if msg_type == "audio":
                    data = msg.get("data") or {}
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        chunk = TtsAudioChunk(
                            sentence_index=self._current_sentence_index,
                            chunk_index=self._chunk_index_counter,
                            format="mp3",
                            audio=audio_b64,
                            is_last=False,
                        )
                        self._chunk_index_counter += 1
                        await self._audio_queue.put(chunk)

                elif msg_type == "event":
                    data = msg.get("data") or {}
                    if data.get("event_type") == "final":
                        # Mark final audio chunk if any or signal finality
                        chunk = TtsAudioChunk(
                            sentence_index=self._current_sentence_index,
                            chunk_index=self._chunk_index_counter,
                            format="mp3",
                            audio="",
                            is_last=True,
                        )
                        await self._audio_queue.put(chunk)

                elif msg_type == "error":
                    data = msg.get("data") or {}
                    err_msg = data.get("message", "Unknown error from Sarvam TTS")
                    logger.error(f"Sarvam streaming TTS error: {err_msg}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading Sarvam TTS WebSocket stream: {e}")
        finally:
            await self._audio_queue.put(None)  # Sentinel indicating stream ended

    async def receive_chunks(self) -> AsyncIterator[TtsAudioChunk]:
        """Yields audio chunks as they arrive from the Sarvam streaming socket."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None or self.is_closed:
                break
            if chunk.audio or chunk.is_last:
                yield chunk

    async def close(self):
        """Immediately closes the WebSocket connection and terminates reader task."""
        if self.is_closed:
            return
        self.is_closed = True
        self.is_connected = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        await self._audio_queue.put(None)
