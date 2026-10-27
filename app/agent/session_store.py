import time
import uuid
from typing import Dict, Optional
from .session import TravelSession, create_initial_session


class SessionStoreService:
    def __init__(self, ttl_seconds: int = 7200):
        self.sessions: Dict[str, TravelSession] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, session_id: Optional[str] = None) -> TravelSession:
        if not session_id:
            session_id = str(uuid.uuid4())
        session = self.sessions.get(session_id)
        if not session:
            session = create_initial_session(session_id)
            self.sessions[session_id] = session
        return session

    def save(self, session: TravelSession) -> TravelSession:
        session.updatedAt = int(time.time() * 1000)
        session.stateVersion += 1
        if len(self.sessions) >= 1000 and session.sessionId not in self.sessions:
            first_key = next(iter(self.sessions))
            del self.sessions[first_key]
        self.sessions[session.sessionId] = session
        return session

    def reset(self, session_id: str) -> TravelSession:
        fresh = create_initial_session(session_id)
        self.sessions[session_id] = fresh
        return fresh

    def cleanup(self):
        now_ms = int(time.time() * 1000)
        ttl_ms = self.ttl_seconds * 1000
        to_del = [sid for sid, s in self.sessions.items() if now_ms - s.updatedAt > ttl_ms]
        for sid in to_del:
            self.sessions.pop(sid, None)


# Global singleton instance
session_store = SessionStoreService()
