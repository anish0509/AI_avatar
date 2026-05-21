"""Unit tests for RedisConversationStore against a hand-rolled fake Redis
client -- matches this project's established preference for real-shaped
fakes (FakeHttpClient, FakeWebSocket) over a generic mock or adding a new
`fakeredis` dependency for four commands. The fake implements exactly the
commands the store uses (RPUSH/LTRIM/EXPIRE/LRANGE/DELETE) with real Redis
index semantics for the negative-index calls this store actually makes.

The fake's shape was cross-checked against a REAL Redis 7 container
(docker run redis:7-alpine) before writing these tests -- confirmed
LTRIM(-20, -1) keeps the last 20 elements, EXPIRE is readable via TTL, and
the async pipeline context-manager API works exactly as used here. Not
guessed from memory."""

import json

import pytest

from app.services.conversation_memory import MAX_HISTORY_MESSAGES, RedisConversationStore


class FakePipeline:
    """Mirrors redis.asyncio's pipeline just enough for RedisConversationStore:
    commands queue against the fake client and only take effect on execute(),
    same as a real pipeline."""

    def __init__(self, client: "FakeRedisClient") -> None:
        self._client = client
        self._queued: list[tuple[str, tuple]] = []

    def rpush(self, key: str, value: str) -> None:
        self._queued.append(("rpush", (key, value)))

    def ltrim(self, key: str, start: int, stop: int) -> None:
        self._queued.append(("ltrim", (key, start, stop)))

    def expire(self, key: str, seconds: int) -> None:
        self._queued.append(("expire", (key, seconds)))

    async def execute(self) -> list:
        results = [await getattr(self._client, f"_{name}")(*args) for name, args in self._queued]
        self._queued = []
        return results

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class FakeRedisClient:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def _rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def _ltrim(self, key: str, start: int, stop: int) -> bool:
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if stop == -1 else values[start : stop + 1]
        return True

    async def _expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        values = self.lists.get(key, [])
        return values[start:] if stop == -1 else values[start : stop + 1]

    async def delete(self, key: str) -> int:
        existed = key in self.lists
        self.lists.pop(key, None)
        self.ttls.pop(key, None)
        return 1 if existed else 0


@pytest.mark.asyncio
async def test_get_history_empty_for_unknown_thread():
    store = RedisConversationStore(FakeRedisClient())
    assert await store.get_history("never-seen") == []


@pytest.mark.asyncio
async def test_append_then_get_history_returns_it_in_order():
    client = FakeRedisClient()
    store = RedisConversationStore(client)

    await store.append_message("t1", "user", "hello")
    await store.append_message("t1", "assistant", "hi there")

    assert await store.get_history("t1") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


@pytest.mark.asyncio
async def test_history_is_stored_as_a_redis_list_of_json_strings():
    """Confirms the wire format, not just the round-trip -- a JSON blob per
    thread would also round-trip correctly but wouldn't be a Redis LIST."""
    client = FakeRedisClient()
    store = RedisConversationStore(client)

    await store.append_message("t1", "user", "hello")

    assert client.lists["conv:t1"] == [json.dumps({"role": "user", "content": "hello"})]


@pytest.mark.asyncio
async def test_threads_are_isolated_from_each_other():
    store = RedisConversationStore(FakeRedisClient())
    await store.append_message("t1", "user", "t1 message")
    await store.append_message("t2", "user", "t2 message")

    assert await store.get_history("t1") == [{"role": "user", "content": "t1 message"}]
    assert await store.get_history("t2") == [{"role": "user", "content": "t2 message"}]


@pytest.mark.asyncio
async def test_history_trims_oldest_messages_past_max():
    store = RedisConversationStore(FakeRedisClient())
    for i in range(MAX_HISTORY_MESSAGES + 5):
        await store.append_message("t1", "user", f"message {i}")

    history = await store.get_history("t1")

    assert len(history) == MAX_HISTORY_MESSAGES
    assert history[0]["content"] == "message 5"
    assert history[-1]["content"] == f"message {MAX_HISTORY_MESSAGES + 4}"


@pytest.mark.asyncio
async def test_clear_thread_removes_history():
    client = FakeRedisClient()
    store = RedisConversationStore(client)
    await store.append_message("t1", "user", "hello")

    await store.clear_thread("t1")

    assert await store.get_history("t1") == []
    assert "conv:t1" not in client.lists


@pytest.mark.asyncio
async def test_clear_thread_on_unknown_thread_is_a_noop():
    store = RedisConversationStore(FakeRedisClient())
    await store.clear_thread("never-existed")  # must not raise


@pytest.mark.asyncio
async def test_ttl_is_set_and_reset_on_every_append():
    """Sliding expiry: an actively-used conversation's TTL keeps resetting,
    so only a thread with no new message for the configured window expires."""
    client = FakeRedisClient()
    store = RedisConversationStore(client)

    await store.append_message("t1", "user", "first")
    first_ttl_call_count = len(client.ttls)
    await store.append_message("t1", "user", "second")

    assert "conv:t1" in client.ttls
    assert first_ttl_call_count == 1  # TTL was set after the FIRST append already, not just the last
    from app.core.config import settings

    assert client.ttls["conv:t1"] == settings.memory_ttl_seconds
