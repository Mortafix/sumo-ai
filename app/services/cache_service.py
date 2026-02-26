import asyncio
import time
from dataclasses import dataclass
from typing import Callable, TypedDict


class CachePayload(TypedDict):
    summary: str
    language: str


class CacheValue(CachePayload):
    created_at: float
    expires_at: float


@dataclass
class CacheEntry:
    summary: str
    language: str
    created_at: float
    expires_at: float


def make_key(video_id: str, mode: str) -> str:
    return f"{video_id}:{mode}"


class InMemoryTTLCache:
    def __init__(self, time_provider: Callable[[], float] | None = None) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._time = time_provider or time.time

    async def get(self, key: str) -> CacheValue | None:
        now = self._time()
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            if now > entry.expires_at:
                self._store.pop(key, None)
                return None

            return {
                "summary": entry.summary,
                "language": entry.language,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
            }

    async def set(
        self,
        key: str,
        value: CachePayload,
        ttl_seconds: int,
    ) -> CacheValue:
        now = self._time()
        expires_at = now + ttl_seconds
        entry = CacheEntry(
            summary=value["summary"],
            language=value["language"],
            created_at=now,
            expires_at=expires_at,
        )
        async with self._lock:
            self._store[key] = entry

        return {
            "summary": entry.summary,
            "language": entry.language,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
        }
