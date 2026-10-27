import re
import base64
import logging
import httpx
from typing import Optional
from ..config import settings

logger = logging.getLogger("tts_service")


class TtsService:
    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.cache: dict[str, bytes] = {}

    async def synthesize(self, text: str, character: str = "meera", voice_id: Optional[str] = None) -> bytes:
        if not text or not text.strip():
            return b""

        cache_key = f"{character}:{voice_id}:{text.strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.api_key:
            logger.warning("SARVAM_API_KEY is not configured, skipping TTS audio synthesis.")
            return b""

        # Bulbul V4 Flash persona mapping:
        is_kabir = (character == "kabir" or voice_id in ("kabir", "kabir_hi_conversational", "aditya"))
        speaker = "kabir_hi_conversational" if is_kabir else "priya_hi_recovery"
        is_hindi = bool(re.search(r"[\u0900-\u097f]", text) or re.search(r"\b(hai|hain|jana|jaana|sasta|sasti|karo|batao|se|ko|parso|kal)\b", text, re.IGNORECASE))
        target_lang = "hi-IN" if is_hindi else "en-IN"

        payload = {
            "text": text,
            "target_language_code": target_lang,
            "speaker": speaker,
            "model": getattr(settings, "SARVAM_TTS_MODEL", "bulbul:v4-flash"),
            "pace": 1.0,
            "temperature": 0.6,
            "speech_sample_rate": 22050,
            "output_audio_codec": "mp3"
        }

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post("https://api.sarvam.ai/text-to-speech", json=payload, headers=headers)
                if resp.status_code != 200:
                    logger.error(f"Sarvam TTS failed: status {resp.status_code}, {resp.text}")
                    return b""
                data = resp.json()

            audios = data.get("audios") or []
            if audios and isinstance(audios[0], str):
                audio_bytes = base64.b64decode(audios[0])
                self.cache[cache_key] = audio_bytes
                return audio_bytes
        except Exception as e:
            logger.error(f"Sarvam TTS error: {e}")

        return b""
