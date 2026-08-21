import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.task_bridge import wait_and_cancel_rest
from app.services.conversation_memory import InMemoryConversationStore


def test_app_starts_without_api_credentials():
    env = {**os.environ, "OPENAI_API_KEY": "", "LOGFIRE_TOKEN": "", "LANGSMITH_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-c", "from app.main import app; from fastapi.testclient import TestClient; assert TestClient(app).get('/health').status_code == 200"],
        env=env, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_task_failure_waits_for_sibling_cleanup():
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def fail():
        await started.wait()
        raise RuntimeError("producer failed")

    async def sibling():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    tasks = (asyncio.create_task(fail()), asyncio.create_task(sibling()))
    try:
        with pytest.raises(RuntimeError, match="producer failed"):
            await wait_and_cancel_rest(tasks, asyncio.FIRST_EXCEPTION)
        assert cleaned.is_set()
        assert all(task.done() for task in tasks)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_history_messages_cannot_mutate_store():
    store = InMemoryConversationStore()
    await store.append_message("thread", "user", "original")
    history = await store.get_history("thread")
    history[0]["content"] = "changed"
    assert await store.get_history("thread") == [{"role": "user", "content": "original"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"prompt": None}, {"prompt": 42}, {"prompt": []}, json.JSONDecodeError("bad json", "{", 1)])
async def test_voice_rejects_invalid_prompt(payload):
    from app.api.voice_routes import voice_ws
    from tests.test_orchestration import FakeBrowserSocket

    socket = FakeBrowserSocket(prompt_payload=payload)
    await voice_ws(socket)
    assert socket.closed
    assert socket.sent[0]["type"] == "error"


@pytest.mark.asyncio
async def test_disconnect_watcher_ignores_non_disconnect_frames():
    from app.api.voice_routes import _watch_for_disconnect
    from fastapi import WebSocketDisconnect

    class Socket:
        def __init__(self):
            self.messages = iter([
                {"type": "websocket.receive", "text": "ping"},
                {"type": "websocket.disconnect", "code": 1000},
            ])

        async def receive(self):
            return next(self.messages)

    with pytest.raises(WebSocketDisconnect):
        await _watch_for_disconnect(Socket())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tts", "stt"])
async def test_failed_realtime_setup_closes_socket(kind, monkeypatch):
    from unittest.mock import AsyncMock
    from app.core.config import settings
    from app.services import realtime_stt, realtime_tts
    from tests.test_realtime_tts import FakeWebSocket

    monkeypatch.setattr(settings, "openai_api_key", "test-placeholder")
    module = realtime_tts if kind == "tts" else realtime_stt
    factory = module.RealtimeApiSpeaker if kind == "tts" else module.RealtimeApiTranscriber
    socket = FakeWebSocket([{"type": "error", "error": {"message": "invalid session"}}])
    monkeypatch.setattr(module.websockets, "connect", AsyncMock(return_value=socket))
    with pytest.raises(RuntimeError, match="Realtime API error"):
        async with factory():
            pass
    assert socket.closed


@pytest.mark.asyncio
async def test_cancelling_bridge_waits_for_children():
    ready = asyncio.Event()
    cleaned = asyncio.Event()

    async def child():
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    task = asyncio.create_task(child())
    bridge = asyncio.create_task(wait_and_cancel_rest((task,), asyncio.FIRST_COMPLETED))
    await ready.wait()
    bridge.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bridge
    assert task.done()
    assert cleaned.is_set()
