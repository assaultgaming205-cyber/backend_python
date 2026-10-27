from typing import Optional
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel
from ..ai.tts_service import TtsService
from ..config import settings

router = APIRouter(tags=["tts"])
tts_service = TtsService()

SARVAM_SPEAKERS = [
    {"id": "priya_hi_recovery", "speakerId": "priya_hi_recovery", "name": "Priya (Meera)", "gender": "female", "language": "Hindi/English", "model": "bulbul:v4-flash"},
    {"id": "kabir_hi_conversational", "speakerId": "kabir_hi_conversational", "name": "Kabir", "gender": "male", "language": "Hindi/English", "model": "bulbul:v4-flash"},
]


class TtsRequest(BaseModel):
    text: str
    character: Optional[str] = "meera"
    speaker: Optional[str] = None
    voice: Optional[str] = None
    gender: Optional[str] = None


@router.get("/tts/voices")
@router.get("/api/tts/voices")
@router.get("/chat/tts/voices")
@router.get("/api/chat/tts/voices")
async def get_voices():
    return {
        "sarvamConfigured": bool(settings.SARVAM_API_KEY),
        "defaultProvider": "sarvam" if settings.SARVAM_API_KEY else "none",
        "sarvamSpeakers": SARVAM_SPEAKERS,
    }


@router.post("/tts/synthesize")
@router.post("/api/tts/synthesize")
@router.post("/chat/tts")
@router.post("/api/chat/tts")
async def synthesize_speech(body: TtsRequest):
    char = body.character or ("kabir" if body.gender == "male" else "meera")
    speaker = body.speaker or body.voice
    audio = await tts_service.synthesize(body.text, character=char, voice_id=speaker)
    if not audio:
        raise HTTPException(status_code=502, detail="TTS synthesis failed or not configured.")

    content_type = "audio/wav" if audio[:4] == b"RIFF" else "audio/mpeg"
    return Response(content=audio, media_type=content_type)
