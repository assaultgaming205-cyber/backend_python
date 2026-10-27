import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    NODE_ENV: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: str = "*"
    APP_TIMEZONE: str = "Asia/Kolkata"

    # AI Configuration (NVIDIA Nemotron)
    AI_PROVIDER: str = "nvidia"
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    NVIDIA_TIMEOUT_MS: int = 45000
    LLM_TOKEN_STREAMING_ENABLED: bool = False

    # STT (Native Android/iOS Device Speech-to-Text — Parakeet & Sarvam STT disabled)
    STT_ENABLED: bool = False
    NVIDIA_ASR_API_KEY: str = ""
    NVIDIA_ASR_GRPC_URL: str = "grpc.nvcf.nvidia.com:443"
    STT_DEFAULT_LANGUAGE: str = "auto"
    STT_TIMEOUT_MS: int = 15000

    # Travel Providers
    FLIGHT_PROVIDER: str = "bankg"
    BANKG_BASE_URL: str = "https://bankg-stagging-api.testuat.com/travel/v1/api"
    BANKG_GATEWAY_URL: str = "https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master-gateway/airSearch"
    ENABLE_FARE_VALIDATE: bool = False
    ENABLE_FLIGHT_BOOK: bool = False  # NON-NEGOTIABLE SAFETY GATE: flightBook permanently blocked
    ENABLE_AMENDMENT_SUBMIT: bool = False  # NON-NEGOTIABLE SAFETY GATE: submitAmendment permanently blocked
    HOTEL_PROVIDER: str = "makemytrip"
    RAIL_PROVIDER: str = "makemytrip"
    TRAVEL_API_ONLY: bool = True
    PROVIDER_TIMEOUT_MS: int = 45000
    FLIGHT_PROVIDER_TIMEOUT_MS: int = 45000

    # MakeMyTrip URLs
    MMT_RAILWAYS_BASE_URL: str = "https://railways.makemytrip.com"
    MMT_RAILWAYS_ALTERNATE_BASE_URL: str = "https://railways.makemytrip.com"
    MMT_FLIGHTS_BASE_URL: str = "https://flights-cb.makemytrip.com/api/fareCalendar"
    MMT_HOTELS_BASE_URL: str = "https://mapi.makemytrip.com/clientbackend/cg/search-hotels"
    MMT_COOKIE: str = ""

    # TTS (Sarvam AI)
    TTS_PROVIDER: str = "sarvam"
    SARVAM_API_KEY: str = ""
    SARVAM_VOICE: str = "priya"
    STREAMING_TTS_ENABLED: bool = False
    BARGE_IN_ENABLED: bool = False
    SARVAM_TTS_MODEL: str = "bulbul:v4-flash"
    SARVAM_TTS_WS_URL: str = "wss://api.sarvam.ai/text-to-speech/ws"

    # STT (Sarvam AI Realtime)
    SARVAM_STT_ENABLED: bool = False
    SARVAM_STT_MODEL: str = "saaras:v3-realtime"
    SARVAM_STT_WS_URL: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    SARVAM_STT_STREAM_TYPE: str = "fast"
    SARVAM_STT_ENDPOINTING: str = "vad"
    SARVAM_STT_SILENCE_DURATION_MS: int = 2000

    @property
    def cors_origin_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
