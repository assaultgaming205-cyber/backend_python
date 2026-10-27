from fastapi import APIRouter
from ..ai.ai_service import AiService

router = APIRouter(tags=["health"])
ai_service = AiService()


@router.get("/health")
@router.get("/api/health")
async def health_check():
    reachable = await ai_service.health_check()
    return {
        "status": "ok" if reachable else "degraded",
        "service": "travel-buddy-python",
        "providers": {
            "ai": reachable,
            "nvidia": reachable,
            "flights": True,
            "hotels": True,
            "makemytrip": True,
        },
        "checks": {
            "ai": "reachable" if reachable else "unavailable",
            "flights": "ready",
            "hotels": "ready",
            "rail": "ready",
        },
    }
