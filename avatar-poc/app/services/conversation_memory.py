"""Per-thread conversation history for the agent (Step 5), backed by a
swappable store (Phase 2, 2026-07-30).

Public surface is unchanged in SHAPE but not in signature: `get_history` /
`append_message` / `clear_thread` are the same three functions callers
already use, now `async def`. That's a real, deliberate change, not a typo --
Redis calls are I/O, and calling a blocking client from these functions
without `await` would be the exact class of bug already found and fixed for
Pinecone (asyncio.to_thread in retrieval.py, P18 in
architecture-and-query-flow.md): a synchronous call silently freezing the
event loop. Every caller (agent.py) is already inside an `async def`, so this
is a mechanical `await` added at each call site, not a structural change.

Two backends behind one `ConversationStore` interface, chosen by
`settings.memory_backend` (mirrors the ABC + concrete-implementation pattern
already used for `TtsSpeaker`/`AvatarRenderer`, with the lazy-singleton
selection itself closer to `vector_store.py`'s `_get_client()`/`ensure_index()`
-- one client built once, reused for the process's lifetime, not a fresh
instance per call):

- InMemoryConversationStore: today's dict, wrapped. Default. What every
  pre-existing test runs against -- restart/multi-worker limitations are
  unchanged and still an accepted POC limitation, not a bug.
- RedisConversationStore: a Redis LIST per thread (RPUSH/LTRIM/LRANGE), not a
  JSON blob -- appending is push-then-trim, never read-modify-write, so there
  is no lost-update race even under concurrent writers. TTL is reset on every
  append (sliding expiry, verified against a real Redis container): an
  actively-used conversation never expires mid-use, only a thread with no new
  message for `settings.memory_ttl_seconds` (24h default) is deleted.

Redis errors are NOT caught here and propagate to the caller. Deliberate,
matching this project's established preference for loud failures over silent
degradation (the whole finding in P18 was that a SILENT failure is the
expensive kind) -- if Redis is unreachable, the turn fails visibly rather than
quietly answering with no memory and no sign anything was wrong.
"""

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
        return list(self._threads.get(thread_id, []))

    async def append_message(self, thread_id: str, role: str, content: str) -> None:
        history = self._threads.setdefault(thread_id, [])
        history.append({"role": role, "content": content})
        if len(history) > MAX_HISTORY_MESSAGES:
            del history[: len(history) - MAX_HISTORY_MESSAGES]

    async def clear_thread(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)


class RedisConversationStore(ConversationStore):
    """Takes an already-constructed client (dependency injection) rather than
    building its own from a URL internally -- lets tests pass a fake client
    directly instead of monkeypatching module internals, same reasoning as
    this project's other fakes (FakeHttpClient, FakeWebSocket)."""

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
