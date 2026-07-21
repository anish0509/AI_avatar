"""Bounded conversation history with in-process and Redis stores. Redis failures propagate to callers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from app.core.config import settings

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

# Keeps prompts (and thus token cost/latency) bounded for long-running
# threads; oldest messages drop first once exceeded.
MAX_HISTORY_MESSAGES = 20


class ConversationStore(ABC):
    @abstractmethod
    async def get_history(self, thread_id: str) -> list[Message]:
        """Returns a copy -- callers must not mutate the stored history directly."""

    @abstractmethod
    async def append_message(self, thread_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    async def clear_thread(self, thread_id: str) -> None: ...


class InMemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        self._threads: dict[str, list[Message]] = {}

    async def get_history(self, thread_id: str) -> list[Message]:
        return [message.copy() for message in self._threads.get(thread_id, [])]

    async def append_message(self, thread_id: str, role: str, content: str) -> None:
        history = self._threads.setdefault(thread_id, [])
        history.append({"role": role, "content": content})
        if len(history) > MAX_HISTORY_MESSAGES:
            del history[: len(history) - MAX_HISTORY_MESSAGES]

    async def clear_thread(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)


class RedisConversationStore(ConversationStore):
    """Store messages in Redis lists with atomic append, trim, and sliding expiry."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def _key(self, thread_id: str) -> str:
        return f"conv:{thread_id}"

    async def get_history(self, thread_id: str) -> list[Message]:
        raw_messages = await self._redis.lrange(self._key(thread_id), 0, -1)
        return [json.loads(m) for m in raw_messages]

    async def append_message(self, thread_id: str, role: str, content: str) -> None:
        key = self._key(thread_id)
        payload = json.dumps({"role": role, "content": content})
        # One round trip, not three: push, trim to the cap, reset the TTL --
        # verified against a real Redis container that LTRIM(-N, -1) keeps
        # the LAST N elements (oldest dropped), matching the in-memory
        # store's behavior exactly.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -MAX_HISTORY_MESSAGES, -1)
            pipe.expire(key, settings.memory_ttl_seconds)
            await pipe.execute()

    async def clear_thread(self, thread_id: str) -> None:
        await self._redis.delete(self._key(thread_id))


_store: ConversationStore | None = None


def _build_store() -> ConversationStore:
    if settings.memory_backend == "redis":
        import redis.asyncio as redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        return RedisConversationStore(client)
    return InMemoryConversationStore()


def _get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = _build_store()
    return _store


async def get_history(thread_id: str) -> list[Message]:
    return await _get_store().get_history(thread_id)


async def append_message(thread_id: str, role: str, content: str) -> None:
    await _get_store().append_message(thread_id, role, content)


async def clear_thread(thread_id: str) -> None:
    await _get_store().clear_thread(thread_id)
