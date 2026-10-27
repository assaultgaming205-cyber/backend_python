from fastapi import APIRouter
from ..config import settings

router = APIRouter(tags=["ai"])


@router.get("/ai/options")
@router.get("/api/ai/options")
async def ai_options():
    return {
        "defaultProvider": "nvidia",
        "defaultModel": settings.NVIDIA_MODEL,
        "models": [
            {
                "id": "nvidia/nemotron-3-ultra-550b-a55b",
                "name": "Nemotron 550B Ultra",
                "shortName": "550B Ultra",
                "tag": "Deep & Fluent",
                "speed": "2–5s",
            },
            {
                "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "name": "Nemotron 30B Lightning",
                "shortName": "30B Lightning",
                "tag": "Ultra Fast",
                "speed": "<1.5s",
            },
        ],
        "providers": [
            {
                "id": "nvidia",
                "label": f"NVIDIA ({settings.NVIDIA_MODEL})",
                "model": settings.NVIDIA_MODEL,
                "configured": bool(settings.NVIDIA_API_KEY),
            }
        ],
    }
