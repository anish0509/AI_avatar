"""Unit tests for app/services/conversation_memory.py's public facade. Runs
against InMemoryConversationStore (the default backend) -- no Redis, no API
calls. Mechanically converted to async in Phase 2 (2026-07-30): the public
functions became `async def` so a Redis-backed store's I/O could be awaited
without freezing the event loop (the same class of bug already found and
fixed for Pinecone, P18) -- these tests' behavior is otherwise unchanged from
before that change, proving the default backend's behavior didn't shift."""

import pytest

from app.services.conversation_memory import (
    MAX_HISTORY_MESSAGES,
    append_message,
    clear_thread,
    get_history,
)


@pytest.mark.asyncio
async def test_get_history_empty_for_unknown_thread():
    assert await get_history("never-seen-thread") == []


@pytest.mark.asyncio
async def test_append_message_then_get_history_returns_it():
    thread = "thread-a"
    await append_message(thread, "user", "hello")
    await append_message(thread, "assistant", "hi there")

    history = await get_history(thread)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_get_history_returns_a_copy_not_the_live_list():
    thread = "thread-b"
    await append_message(thread, "user", "hello")

    history = await get_history(thread)
    history.append({"role": "user", "content": "mutated externally"})

    assert await get_history(thread) == [{"role": "user", "content": "hello"}]
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_threads_are_isolated_from_each_other():
    await append_message("thread-c1", "user", "c1 message")
    await append_message("thread-c2", "user", "c2 message")

    assert await get_history("thread-c1") == [{"role": "user", "content": "c1 message"}]
    assert await get_history("thread-c2") == [{"role": "user", "content": "c2 message"}]
    await clear_thread("thread-c1")
    await clear_thread("thread-c2")


@pytest.mark.asyncio
async def test_history_trims_oldest_messages_past_max():
    thread = "thread-d"
    for i in range(MAX_HISTORY_MESSAGES + 5):
        await append_message(thread, "user", f"message {i}")

    history = await get_history(thread)

    assert len(history) == MAX_HISTORY_MESSAGES
    # oldest messages (0-4) were dropped; the tail survives in order.
    assert history[0]["content"] == "message 5"
    assert history[-1]["content"] == f"message {MAX_HISTORY_MESSAGES + 4}"
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_clear_thread_removes_history():
    thread = "thread-e"
    await append_message(thread, "user", "hello")

    await clear_thread(thread)

    assert await get_history(thread) == []


@pytest.mark.asyncio
async def test_clear_thread_on_unknown_thread_is_a_noop():
    await clear_thread("never-existed")  # must not raise


# ── backend selection (_build_store / _get_store) ──────────────────────────


def test_build_store_defaults_to_in_memory(monkeypatch):
    from app.services import conversation_memory as mem

    monkeypatch.setattr(mem.settings, "memory_backend", "memory")
    assert isinstance(mem._build_store(), mem.InMemoryConversationStore)


def test_build_store_selects_redis_when_configured(monkeypatch):
    from app.services import conversation_memory as mem

    monkeypatch.setattr(mem.settings, "memory_backend", "redis")
    # redis.asyncio.from_url() builds a client lazily -- it does not connect
    # until a command is actually sent, so this doesn't require a running
    # Redis (confirmed: no network call happens during construction).
    store = mem._build_store()
    assert isinstance(store, mem.RedisConversationStore)


def test_get_store_caches_the_singleton(monkeypatch):
    from app.services import conversation_memory as mem

    monkeypatch.setattr(mem, "_store", None)
    first = mem._get_store()
    second = mem._get_store()
    assert first is second
    monkeypatch.setattr(mem, "_store", None)  # don't leak into other tests
