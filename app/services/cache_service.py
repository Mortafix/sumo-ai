import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, TypedDict


class CachePayloadBase(TypedDict):
    summary: str
    language: str


class CachePayload(CachePayloadBase, total=False):
    transcript: str


class CacheValue(CachePayload, total=False):
    created_at: float
    expires_at: float


@dataclass
class CacheEntry:
    summary: str
    language: str
    transcript: str | None
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

            value: CacheValue = {
                "summary": entry.summary,
                "language": entry.language,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
            }
            if entry.transcript is not None:
                value["transcript"] = entry.transcript
            return value

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
            transcript=value.get("transcript"),
            created_at=now,
            expires_at=expires_at,
        )
        async with self._lock:
            self._store[key] = entry

        payload: CacheValue = {
            "summary": entry.summary,
            "language": entry.language,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
        }
        if entry.transcript is not None:
            payload["transcript"] = entry.transcript
        return payload


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatSessionPayload(TypedDict):
    video_id: str
    mode: str
    language: str
    summary: str
    transcript: str
    processing_ms: float


class ChatSessionValue(ChatSessionPayload):
    chat_id: str
    active_form_token: str
    history: list[ChatMessage]
    user_messages_count: int
    created_at: float
    expires_at: float


@dataclass
class ChatSessionEntry:
    chat_id: str
    active_form_token: str
    video_id: str
    mode: str
    language: str
    summary: str
    transcript: str
    processing_ms: float
    history: list[ChatMessage] = field(default_factory=list)
    user_messages_count: int = 0
    created_at: float = 0.0
    expires_at: float = 0.0


class InMemoryChatSessionStore:
    def __init__(self, time_provider: Callable[[], float] | None = None) -> None:
        self._store: dict[str, ChatSessionEntry] = {}
        self._lock = asyncio.Lock()
        self._time = time_provider or time.time

    @staticmethod
    def _copy_history(history: list[ChatMessage]) -> list[ChatMessage]:
        return [{"role": item["role"], "content": item["content"]} for item in history]

    def _to_value(self, entry: ChatSessionEntry) -> ChatSessionValue:
        return {
            "chat_id": entry.chat_id,
            "active_form_token": entry.active_form_token,
            "video_id": entry.video_id,
            "mode": entry.mode,
            "language": entry.language,
            "summary": entry.summary,
            "transcript": entry.transcript,
            "processing_ms": entry.processing_ms,
            "history": self._copy_history(entry.history),
            "user_messages_count": entry.user_messages_count,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
        }

    async def create(
        self, payload: ChatSessionPayload, ttl_seconds: int
    ) -> ChatSessionValue:
        now = self._time()
        expires_at = now + ttl_seconds
        async with self._lock:
            chat_id = secrets.token_urlsafe(18)
            while chat_id in self._store:
                chat_id = secrets.token_urlsafe(18)
            entry = ChatSessionEntry(
                chat_id=chat_id,
                active_form_token=secrets.token_urlsafe(12),
                video_id=payload["video_id"],
                mode=payload["mode"],
                language=payload["language"],
                summary=payload["summary"],
                transcript=payload["transcript"],
                processing_ms=payload["processing_ms"],
                created_at=now,
                expires_at=expires_at,
            )
            self._store[chat_id] = entry
            return self._to_value(entry)

    async def get(self, chat_id: str) -> ChatSessionValue | None:
        now = self._time()
        async with self._lock:
            entry = self._store.get(chat_id)
            if entry is None:
                return None
            if now > entry.expires_at:
                self._store.pop(chat_id, None)
                return None
            return self._to_value(entry)

    async def verify_form_token(self, chat_id: str, token: str) -> ChatSessionValue:
        now = self._time()
        normalized_token = (token or "").strip()
        async with self._lock:
            entry = self._store.get(chat_id)
            if entry is None:
                raise KeyError(chat_id)
            if now > entry.expires_at:
                self._store.pop(chat_id, None)
                raise KeyError(chat_id)
            if not normalized_token or normalized_token != entry.active_form_token:
                raise ValueError("Token non valido.")

            entry.active_form_token = secrets.token_urlsafe(12)
            return self._to_value(entry)

    async def record_exchange(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
        max_user_messages: int,
    ) -> ChatSessionValue:
        now = self._time()
        async with self._lock:
            entry = self._store.get(chat_id)
            if entry is None:
                raise KeyError(chat_id)
            if now > entry.expires_at:
                self._store.pop(chat_id, None)
                raise KeyError(chat_id)
            if entry.user_messages_count >= max_user_messages:
                raise ValueError("Limite messaggi raggiunto.")

            entry.history.append({"role": "user", "content": user_message})
            entry.history.append({"role": "assistant", "content": assistant_message})
            entry.user_messages_count += 1
            return self._to_value(entry)
