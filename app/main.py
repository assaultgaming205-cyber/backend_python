import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.health import router as health_router
from .api.chat import router as chat_router
from .api.flights import router as flights_router
from .api.trains import router as trains_router
from .api.hotels import router as hotels_router
from .api.tts import router as tts_router
from .api.stt import router as stt_router
from .api.ai import router as ai_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("travel_buddy_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Travel Buddy Python backend...")
    logger.info("Config loaded")
    yield
    logger.info("Shutting down Travel Buddy Python backend...")


app = FastAPI(
    title="Travel Buddy API (Python)",
    description="Conversational AI travel companion backend in Python (FastAPI + Pydantic v2 + Playwright + NVIDIA Nemotron)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(flights_router)
app.include_router(trains_router)
app.include_router(hotels_router)
app.include_router(tts_router)
app.include_router(stt_router)
app.include_router(ai_router)


@app.get("/")
async def root():
    return {
        "service": "travel-buddy-python",
        "status": "online",
        "docs": "/docs",
    }
