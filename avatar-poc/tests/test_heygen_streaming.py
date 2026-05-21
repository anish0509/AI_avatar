import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import websockets.exceptions

from app.services.heygen_streaming import (
    LiveAvatarRenderer,
    _raise_for_status_with_body,
    build_speak_end_message,
    build_speak_message,
    build_token_request,
)

TOKEN_RESPONSE = {"code": 100, "data": {"session_id": "sess-1", "session_token": "tok-1"}, "message": "ok"}


def _start_response(ws_url: "str | None" = "wss://fake/ws") -> dict:
    data = {
        "session_id": "sess-1",
        "livekit_url": "wss://fake/livekit",
        "livekit_client_token": "lk-token",
        "max_session_duration": 600,
    }
    if ws_url is not None:
        data["ws_url"] = ws_url
    return {"code": 100, "data": data, "message": "ok"}


class FakeHttpResponse:
    def __init__(self, json_body: dict) -> None:
        self._json_body = json_body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json_body


class FakeHttpClient:
    """Stands in for httpx.AsyncClient: records every request and replays
    canned responses in the order .post() is called. A queued item that's
    already an httpx.Response (rather than a plain dict) is returned as-is --
    used for error-path tests, where a REAL httpx.Response is needed so
    .raise_for_status() raises a real httpx.HTTPStatusError, not a fake one."""

    def __init__(self, responses: "list[dict | httpx.Response]") -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.closed = False

    async def post(
        self, path: str, headers: "dict | None" = None, json: "dict | None" = None
    ) -> "FakeHttpResponse | httpx.Response":
        self.requests.append({"path": path, "headers": headers, "json": json})
        item = self._responses.pop(0)
        if isinstance(item, httpx.Response):
            return item
        return FakeHttpResponse(item)

    async def aclose(self) -> None:
        self.closed = True


def _error_response(status_code: int, body_text: str) -> httpx.Response:
    request = httpx.Request("POST", "https://api.liveavatar.com/v1/sessions/start")
    return httpx.Response(status_code, request=request, text=body_text)


class FakeWebSocket:
    """Mirrors test_realtime_tts.py's FakeWebSocket: records sent frames,
    replays a canned queue of incoming events on iteration.

    block_when_empty makes the iterator behave like a REAL socket -- awaiting
    the next frame forever instead of ending. Without it, an exhausted queue
    raises StopAsyncIteration, which is exactly why the old inline
    wait-for-acknowledgement bug never showed up here: the fake ended the
    loop instantly where the real socket sat for the full 30s timeout.
    push() delivers an event mid-test, for asserting on what happens while
    HeyGen is still rendering."""

    def __init__(
        self,
        incoming: "list[dict] | None" = None,
        fail_send_after: "int | None" = None,
        block_when_empty: bool = False,
    ) -> None:
        self._incoming = list(incoming or [])
        self._pushed: "asyncio.Queue[dict]" = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False
        self._fail_send_after = fail_send_after
        self._block_when_empty = block_when_empty

    def push(self, event: dict) -> None:
        self._pushed.put_nowait(event)

    async def send(self, raw: str) -> None:
        if self._fail_send_after is not None and len(self.sent) >= self._fail_send_after:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        self.sent.append(json.loads(raw))

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        if self._incoming:
            return json.dumps(self._incoming.pop(0))
        if self._block_when_empty:
            # Blocks until push() or forever -- cancelled on __aexit__.
            return json.dumps(await self._pushed.get())
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


def _patched(http_client: FakeHttpClient, fake_ws: FakeWebSocket):
    return (
        patch("app.services.heygen_streaming.httpx.AsyncClient", lambda *a, **kw: http_client),
        patch("app.services.heygen_streaming.websockets.connect", AsyncMock(return_value=fake_ws)),
    )


async def _achunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def test_build_token_request() -> None:
    req = build_token_request("avatar-1", True, "high", "H264")
    assert req == {
        "mode": "LITE",
        "avatar_id": "avatar-1",
        "is_sandbox": True,
        "video_settings": {"quality": "high", "encoding": "H264"},
    }


def test_build_speak_messages() -> None:
    assert build_speak_message("YWJj", "evt-1") == {"type": "agent.speak", "audio": "YWJj", "event_id": "evt-1"}
    assert build_speak_end_message("evt-1") == {"type": "agent.speak_end", "event_id": "evt-1"}


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "")
    with pytest.raises(RuntimeError, match="HEYGEN_API_KEY"):
        async with LiveAvatarRenderer():
            pass


@pytest.mark.asyncio
async def test_session_start_calls_token_then_start_with_correct_auth(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    # __aexit__ also fires a /sessions/stop call, hence 3 canned responses --
    # see test_exit_stops_session_and_closes_websocket for that call itself.
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.state_updated", "state": "connected"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            assert renderer.session.session_id == "sess-1"
            assert renderer.session.livekit_url == "wss://fake/livekit"
            assert renderer.session.livekit_client_token == "lk-token"

    token_call, start_call, _stop_call = http_client.requests
    assert token_call["path"] == "/sessions/token"
    assert token_call["headers"] == {"X-API-KEY": "test-key"}
    assert token_call["json"]["mode"] == "LITE"
    assert start_call["path"] == "/sessions/start"
    assert start_call["headers"] == {"Authorization": "Bearer tok-1"}


def test_raise_for_status_with_body_includes_the_response_text() -> None:
    resp = _error_response(403, '{"code": 40300, "message": "avatar_id not found for this account"}')
    with pytest.raises(httpx.HTTPStatusError, match="avatar_id not found for this account"):
        _raise_for_status_with_body(resp)


def test_raise_for_status_with_body_is_silent_on_success() -> None:
    resp = httpx.Response(200, request=httpx.Request("POST", "https://api.liveavatar.com/v1/sessions/token"))
    _raise_for_status_with_body(resp)  # must not raise


@pytest.mark.asyncio
async def test_session_start_403_surfaces_heygens_actual_reason(monkeypatch) -> None:
    """Regression: httpx's default HTTPStatusError message is only the status
    line and URL -- a real 403 from a mismatched avatar_id/sandbox combo
    showed up with no way to tell why. This is the fix, proven against a REAL
    httpx.Response so a real HTTPStatusError is raised, not a stubbed one."""
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([_error_response(403, "avatar_id is sandbox-only; set is_sandbox=true")])
    fake_ws = FakeWebSocket()

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        with pytest.raises(httpx.HTTPStatusError, match="avatar_id is sandbox-only"):
            async with LiveAvatarRenderer():
                pass


@pytest.mark.asyncio
async def test_missing_ws_url_raises(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(ws_url=None)])
    fake_ws = FakeWebSocket()

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        with pytest.raises(RuntimeError, match="no ws_url"):
            async with LiveAvatarRenderer():
                pass


@pytest.mark.asyncio
async def test_session_stopped_during_connect_raises(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response()])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.stopped", "stop_reason": "NO_CREDITS"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        with pytest.raises(RuntimeError, match="session stopped"):
            async with LiveAvatarRenderer():
                pass


class _FixedUuid:
    hex = "fixed-event-id"


@pytest.mark.asyncio
async def test_speak_audio_sends_chunks_then_speak_end(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response()])
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "session.state_updated", "state": "connected"},
            {"type": "agent.speak_started", "event_id": "fixed-event-id"},
        ]
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2, patch("app.services.heygen_streaming.uuid.uuid4", return_value=_FixedUuid()):
        async with LiveAvatarRenderer() as renderer:
            await renderer.speak_audio(_achunks([b"one", b"two"]))

    speak_frames = [m for m in fake_ws.sent if m["type"] == "agent.speak"]
    end_frames = [m for m in fake_ws.sent if m["type"] == "agent.speak_end"]
    assert len(speak_frames) == 2
    assert len(end_frames) == 1
    assert base64.b64decode(speak_frames[0]["audio"]) == b"one"
    assert base64.b64decode(speak_frames[1]["audio"]) == b"two"
    # same event_id ties both speak frames and the speak_end together
    event_ids = {m["event_id"] for m in speak_frames} | {end_frames[0]["event_id"]}
    assert len(event_ids) == 1


@pytest.mark.asyncio
async def test_speak_audio_returns_without_waiting_for_acknowledgement(monkeypatch) -> None:
    """Regression: speak_audio() used to end by awaiting this event_id's
    agent.speak_started, so sentence N+1 couldn't be sent until HeyGen
    acknowledged sentence N -- and when no acknowledgement ever matched,
    every sentence burned the full heygen_session_timeout_s (30s). A turn
    measured at ~17s over /ws/voice took 91s over /ws/avatar for exactly
    this reason. Here HeyGen sends NO agent.speak_started at all and the
    socket blocks like a real one, so the old code would hang for 30s."""
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(
        incoming=[{"type": "session.state_updated", "state": "connected"}], block_when_empty=True
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            # Well under settings.heygen_session_timeout_s (30.0), so this
            # fails loudly if the inline wait is ever reintroduced.
            async with asyncio.timeout(2.0):
                await renderer.speak_audio(_achunks([b"one"]))
                await renderer.speak_audio(_achunks([b"two"]))

    assert [m["type"] for m in fake_ws.sent] == [
        "agent.speak",
        "agent.speak_end",
        "agent.speak",
        "agent.speak_end",
    ]


@pytest.mark.asyncio
async def test_wait_until_done_blocks_until_every_sentence_has_rendered(monkeypatch) -> None:
    """Regression: speak_audio() returns as soon as the bytes are out, so
    without an explicit drain the route's finally block stopped the HeyGen
    session while the avatar was still speaking -- observed live as the
    answer cutting off mid-sentence with the video going black.

    The pushed events use HeyGen's REAL shape, captured from the live API by
    scripts/check_heygen_session.py: a fresh event_id HeyGen minted itself
    (never the one we sent) plus a task.id. If this code ever goes back to
    correlating on our own event_id, these won't match and the test hangs."""
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(
        incoming=[{"type": "session.state_updated", "state": "connected"}], block_when_empty=True
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            await renderer.speak_audio(_achunks([b"one"]))
            await renderer.speak_audio(_achunks([b"two"]))

            waiter = asyncio.create_task(renderer.wait_until_done())
            await asyncio.sleep(0.05)
            assert not waiter.done()  # nothing rendered yet

            fake_ws.push({"type": "agent.speak_ended", "event_id": "heygen-own-1", "task": {"id": "t1"}})
            await asyncio.sleep(0.05)
            assert not waiter.done()  # only 1 of 2 sentences rendered

            fake_ws.push({"type": "agent.speak_ended", "event_id": "heygen-own-2", "task": {"id": "t2"}})
            async with asyncio.timeout(2.0):
                await waiter


@pytest.mark.asyncio
async def test_wait_until_done_returns_immediately_when_nothing_was_spoken(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(
        incoming=[{"type": "session.state_updated", "state": "connected"}], block_when_empty=True
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            async with asyncio.timeout(2.0):
                await renderer.wait_until_done()


@pytest.mark.asyncio
async def test_wait_until_done_gives_up_once_the_socket_is_gone(monkeypatch) -> None:
    """No block_when_empty, so the reader hits StopAsyncIteration and exits:
    no agent.speak_ended can ever arrive. Without the reader-finished check
    this would sit for the session's whole max_session_duration."""
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.state_updated", "state": "connected"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            await renderer.speak_audio(_achunks([b"one"]))
            await asyncio.sleep(0.01)  # let the reader task finish
            async with asyncio.timeout(2.0):
                await renderer.wait_until_done()


@pytest.mark.asyncio
async def test_session_stopped_between_sentences_surfaces_on_next_speak(monkeypatch) -> None:
    """The background reader is the only thing watching the socket between
    sentences now, so a session HeyGen kills while we're not speaking has to
    be noticed there and re-raised from the next speak_audio() -- an
    exception thrown inside a background task would have nowhere to go."""
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "session.state_updated", "state": "connected"},
            {"type": "session.stopped", "stop_reason": "NO_CREDITS"},
        ],
        block_when_empty=True,
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            await asyncio.sleep(0.01)  # let the reader task drain session.stopped
            with pytest.raises(RuntimeError, match="NO_CREDITS"):
                await renderer.speak_audio(_achunks([b"one"]))


@pytest.mark.asyncio
async def test_speak_audio_raises_clear_error_when_connection_closes_mid_turn(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response()])
    fake_ws = FakeWebSocket(
        incoming=[{"type": "session.state_updated", "state": "connected"}],
        # first send (agent.speak for the one chunk) succeeds, then the
        # connection is closed -- simulates a HeyGen sandbox session
        # expiring (~1 minute) mid-turn, e.g. websockets.exceptions.ConnectionClosedOK
        fail_send_after=1,
    )

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            chunks = _achunks([b"one", b"two"])
            with pytest.raises(RuntimeError, match="sandbox sessions auto-expire"):
                await renderer.speak_audio(chunks)


@pytest.mark.asyncio
async def test_speak_audio_with_no_chunks_sends_nothing(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response()])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.state_updated", "state": "connected"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            await renderer.speak_audio(_achunks([]))

    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_interrupt_sends_agent_interrupt(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response()])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.state_updated", "state": "connected"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer() as renderer:
            await renderer.interrupt()

    assert fake_ws.sent[-1] == {"type": "agent.interrupt"}


@pytest.mark.asyncio
async def test_exit_stops_session_and_closes_websocket(monkeypatch) -> None:
    monkeypatch.setattr("app.services.heygen_streaming.settings.heygen_api_key", "test-key")
    http_client = FakeHttpClient([TOKEN_RESPONSE, _start_response(), {"code": 100, "data": None, "message": "ok"}])
    fake_ws = FakeWebSocket(incoming=[{"type": "session.state_updated", "state": "connected"}])

    p1, p2 = _patched(http_client, fake_ws)
    with p1, p2:
        async with LiveAvatarRenderer():
            pass

    assert fake_ws.closed is True
    assert http_client.closed is True
    stop_call = http_client.requests[-1]
    assert stop_call["path"] == "/sessions/stop"
    assert stop_call["json"] == {"session_id": "sess-1", "reason": "USER_CLOSED"}


@pytest.mark.asyncio
async def test_speak_audio_outside_context_manager_raises() -> None:
    renderer = LiveAvatarRenderer()
    with pytest.raises(RuntimeError, match="outside"):
        await renderer.speak_audio(_achunks([b"x"]))


@pytest.mark.asyncio
async def test_interrupt_outside_context_manager_raises() -> None:
    renderer = LiveAvatarRenderer()
    with pytest.raises(RuntimeError, match="outside"):
        await renderer.interrupt()


def test_session_property_outside_context_manager_raises() -> None:
    renderer = LiveAvatarRenderer()
    with pytest.raises(RuntimeError, match="outside"):
        _ = renderer.session
