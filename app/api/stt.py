import logging
from typing import Optional
from fastapi import APIRouter, Request, WebSocket
from pydantic import BaseModel

from ..config import settings
from ..agent.intent.spoken_normalizer import normalize_spoken_travel_text

logger = logging.getLogger("stt")
router = APIRouter(tags=["stt"])

STT_LANGUAGES = [
    {"code": c, "name": n}
    for c, n in [
        ("auto", "Auto"),
        ("hi", "Hindi"),
        ("en", "English"),
        ("mr", "Marathi"),
        ("gu", "Gujarati"),
        ("ta", "Tamil"),
        ("te", "Telugu"),
        ("bn", "Bengali"),
        ("kn", "Kannada"),
        ("ml", "Malayalam"),
        ("pa", "Punjabi"),
        ("ur", "Urdu"),
    ]
]


class NativeTranscriptRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"


@router.get("/stt/options")
@router.get("/api/stt/options")
async def stt_options():
    """Returns STT configuration. Cloud STT (Parakeet / Sarvam) is disabled so clients use native Android/iOS STT."""
    return {
        "available": False,
        "provider": "device-native",
        "streaming": False,
        "sampleRate": 16000,
        "encoding": "pcm16le",
        "maxSeconds": 60,
        "silenceDurationMs": 2000,
        "defaultLanguage": settings.STT_DEFAULT_LANGUAGE,
        "engines": {
            "device": {
                "model": "android-native-speech",
                "mode": "on-device-realtime",
                "silenceDurationMs": 2000,
            }
        },
        "languages": STT_LANGUAGES,
        "streamPath": "/api/stt/stream",
    }


@router.post("/stt/transcribe")
@router.post("/api/stt/transcribe")
async def stt_transcribe(request: Request, language: Optional[str] = "auto"):
    """Accepts native device transcription text and returns normalized travel text."""
    try:
        data = await request.json()
        raw_text = str(data.get("text") or "").strip()
    except Exception:
        body = await request.body()
        raw_text = body.decode("utf-8", errors="ignore").strip()

    normalized = normalize_spoken_travel_text(raw_text, settings.APP_TIMEZONE) if raw_text else ""
    return {
        "text": normalized or raw_text,
        "rawText": raw_text,
        "language": language or "auto",
        "engine": "device-native",
    }


@router.websocket("/api/stt/stream")
@router.websocket("/stt/stream")
async def stt_stream(ws: WebSocket):
    """Cloud WebSocket STT (Parakeet / Sarvam) is disabled; instructs client to use Android native STT."""
    await ws.accept()
    logger.info("STT WebSocket connected — redirecting client to Android native STT")
    await ws.send_json({
        "type": "error",
        "code": "use_device_stt",
        "message": "Cloud STT (Parakeet/Sarvam) is disabled. Use native Android speech recognition.",
    })
    await ws.close()
