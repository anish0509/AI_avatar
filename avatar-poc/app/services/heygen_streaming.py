"""Drive LiveAvatar lip sync with PCM16 audio. A background reader handles session events while audio is sent."""

import asyncio
import base64
import contextlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import logfire
import websockets

from app.core.config import settings
from app.core.logger import get_logger
from app.core.timing import step_timer

logger = get_logger(__name__)

HEYGEN_SAMPLE_RATE = 24000  # PCM16 mono -- matches TtsSpeaker output exactly
# Session idle timeout is 5 minutes (per LiveAvatar docs); keep well under it.
KEEP_ALIVE_INTERVAL_S = 240
# Fallback only -- the real bound is the session's own max_session_duration,
# which the API reports on /sessions/start (60s for sandbox).
DEFAULT_DRAIN_TIMEOUT_S = 120.0
DRAIN_POLL_INTERVAL_S = 0.1


async def _cancel_task(task: "asyncio.Task | None") -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _raise_for_status_with_body(resp: httpx.Response) -> None:
    """httpx.Response.raise_for_status()'s own error message is only ever the
    status line and URL -- it never includes the response BODY, which is
    where HeyGen's actual reason lives (e.g. "avatar_id not found for this
    account" vs "insufficient credits" both surface as a bare 403/404
    otherwise). Confirmed the hard way: a real 403 from /sessions/start gave
    no way to tell why without this."""
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"{exc}\nHeyGen response body: {resp.text}", request=exc.request, response=exc.response
        ) from exc


def build_token_request(avatar_id: str, is_sandbox: bool, video_quality: str, video_encoding: str) -> dict:
    return {
        "mode": "LITE",
        "avatar_id": avatar_id,
        "is_sandbox": is_sandbox,
        "video_settings": {"quality": video_quality, "encoding": video_encoding},
    }


def build_speak_message(audio_b64: str, event_id: str) -> dict:
    return {"type": "agent.speak", "audio": audio_b64, "event_id": event_id}


def build_speak_end_message(event_id: str) -> dict:
    return {"type": "agent.speak_end", "event_id": event_id}


@dataclass
class AvatarSession:
    """What starting a session returns -- the fields the browser needs to
    view the avatar (livekit_*) plus the ones only the backend uses (ws_url,
    session_id). The HeyGen API key never appears here."""

    session_id: str
    ws_url: str
    livekit_url: str
    livekit_client_token: str
    max_session_duration: int | None


class AvatarRenderer(ABC):
    async def __aenter__(self) -> "AvatarRenderer":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @property
    @abstractmethod
    def session(self) -> AvatarSession:
        """The active session's browser-facing connection details."""

    @abstractmethod
    async def speak_audio(self, pcm_chunks: AsyncIterator[bytes]) -> None:
        """Drive the avatar's lip-sync from a stream of raw PCM16/24kHz mono
        audio chunks -- e.g. a TtsSpeaker.speak() call for one sentence."""

    @abstractmethod
    async def wait_until_done(self) -> None:
        """Block until the avatar has finished rendering everything already
        pushed via speak_audio(), so a caller can tear the session down
        without cutting the avatar off mid-answer."""

    @abstractmethod
    async def interrupt(self) -> None:
        """Stop the avatar mid-speech (e.g. the user cut in)."""


class LiveAvatarRenderer(AvatarRenderer):
    def __init__(self, avatar_id: str | None = None) -> None:
        self._avatar_id = avatar_id or settings.heygen_avatar_id
        self._http: httpx.AsyncClient | None = None
        self._ws = None
        self._session: AvatarSession | None = None
        self._keep_alive_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        # Set on the first audio chunk of the turn, read by the background
        # reader to time the first agent.speak_started against it.
        self._first_speak_at: float | None = None
        self._ttfv_logged = False
        # One "turn" per sentence: incremented here when we send agent.speak_end,
        # and by the reader when HeyGen answers with agent.speak_ended. Equal
        # counts mean the avatar has rendered everything we pushed.
        self._speak_turns_sent = 0
        self._speak_turns_ended = 0
        # Populated by the reader if HeyGen ends the session; surfaced on the
        # next speak_audio() rather than raised from a background task.
        self._session_error: str | None = None

    @property
    def session(self) -> AvatarSession:
        if self._session is None:
            raise RuntimeError("LiveAvatarRenderer used outside 'async with' -- no active session")
        return self._session

    async def __aenter__(self) -> "LiveAvatarRenderer":
        if not settings.heygen_api_key:
            raise RuntimeError("HEYGEN_API_KEY is not set in .env")

        with step_timer("heygen session setup", "heygen_streaming") as timer:
            self._http = httpx.AsyncClient(
                base_url=settings.heygen_api_base, timeout=settings.heygen_session_timeout_s
            )

            token_resp = await self._http.post(
                "/sessions/token",
                headers={"X-API-KEY": settings.heygen_api_key},
                json=build_token_request(
                    self._avatar_id,
                    settings.heygen_is_sandbox,
                    settings.heygen_video_quality,
                    settings.heygen_video_encoding,
                ),
            )
            _raise_for_status_with_body(token_resp)
            session_token = token_resp.json()["data"]["session_token"]
            timer.mark("token")

            start_resp = await self._http.post(
                "/sessions/start", headers={"Authorization": f"Bearer {session_token}"}
            )
            _raise_for_status_with_body(start_resp)
            data = start_resp.json()["data"]
            timer.mark("start")

            ws_url = data.get("ws_url")
            if not ws_url:
                raise RuntimeError(
                    "HeyGen /sessions/start returned no ws_url for a LITE session -- "
                    "backend-driven audio push isn't available; see the Fallback "
                    "section of the integration plan for the LiveKit-based alternative"
                )

            self._session = AvatarSession(
                session_id=data["session_id"],
                ws_url=ws_url,
                livekit_url=data["livekit_url"],
                livekit_client_token=data["livekit_client_token"],
                max_session_duration=data.get("max_session_duration"),
            )

            self._ws = await websockets.connect(ws_url)
            timer.mark("ws_connect")

            # Consumes events inline until "connected"; only after that does the
            # background reader take over as the socket's sole consumer.
            await self._wait_for_state("connected")
            timer.mark("connected_wait")

            self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())
            self._reader_task = asyncio.create_task(self._read_events_loop())
            timer.log(session_id=self._session.session_id)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await _cancel_task(self._keep_alive_task)
        self._keep_alive_task = None
        await _cancel_task(self._reader_task)
        self._reader_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        if self._http is not None and self._session is not None:
            with contextlib.suppress(Exception):
                await self._http.post(
                    "/sessions/stop",
                    headers={"X-API-KEY": settings.heygen_api_key},
                    json={"session_id": self._session.session_id, "reason": "USER_CLOSED"},
                )
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._session = None

    async def speak_audio(self, pcm_chunks: AsyncIterator[bytes]) -> None:
        """Push one sentence's audio and return as soon as the bytes are out.

        Deliberately does NOT wait for HeyGen to acknowledge the speech. It
        used to end by awaiting this event_id's agent.speak_started, which put
        a purely observational latency marker on the critical path: sentence
        N+1 could not be sent until HeyGen acknowledged sentence N, and when
        the acknowledgement never matched, EVERY sentence burned the full
        heygen_session_timeout_s. Measured in a real trace: a turn that takes
        ~17s over /ws/voice took 91s over /ws/avatar -- 3 sentences x 30s of
        dead waiting. The marker is still recorded, just off the hot path, by
        _read_events_loop().
        """
        if self._ws is None:
            raise RuntimeError("LiveAvatarRenderer used outside 'async with' -- no open connection")
        if self._session_error is not None:
            raise RuntimeError(self._session_error)

        event_id = uuid.uuid4().hex
        sent_any = False
        try:
            async for chunk in pcm_chunks:
                if self._first_speak_at is None:
                    self._first_speak_at = time.monotonic()
                audio_b64 = base64.b64encode(chunk).decode("ascii")
                await self._ws.send(json.dumps(build_speak_message(audio_b64, event_id)))
                sent_any = True
            if not sent_any:
                return
            await self._ws.send(json.dumps(build_speak_end_message(event_id)))
            self._speak_turns_sent += 1
        except websockets.exceptions.ConnectionClosed as exc:
            # Sandbox sessions auto-expire after ~1 minute (see docs.liveavatar.com's
            # sandbox-mode page); a multi-sentence answer sent one speak_audio() call
            # per sentence can outlive that window mid-turn. Surface that plainly
            # instead of a raw ConnectionClosedOK traceback.
            raise RuntimeError(
                "HeyGen session closed before this turn finished -- if HEYGEN_IS_SANDBOX=true, "
                "sandbox sessions auto-expire after ~1 minute and a long/multi-sentence answer can "
                "outlive that window. Try a shorter question, or set HEYGEN_IS_SANDBOX=false with a "
                "real avatar_id to remove the limit."
            ) from exc

    async def wait_until_done(self) -> None:
        """Block until HeyGen has rendered every sentence already pushed.

        Necessary because speak_audio() returns as soon as the bytes are out:
        without this the caller's teardown (__aexit__ -> /sessions/stop) fires
        while HeyGen still has queued audio, and the avatar is cut off
        mid-answer with the video going black. The old inline
        wait-for-acknowledgement used to provide this hold accidentally, as a
        side effect of blocking.

        Counts agent.speak_ended against the agent.speak_end frames we sent,
        rather than waiting for agent.idle_started, because HeyGen briefly
        goes idle BETWEEN sentences whenever it drains faster than TTS
        produces -- idle is not a reliable "the whole answer is finished"
        signal, but the counts are.
        """
        if self._speak_turns_sent == 0:
            return
        # No point waiting past the lifetime the API itself gave this session.
        max_duration = self._session.max_session_duration if self._session else None
        timeout_s = float(max_duration or DEFAULT_DRAIN_TIMEOUT_S)
        try:
            async with asyncio.timeout(timeout_s):
                while self._speak_turns_ended < self._speak_turns_sent:
                    if self._reader_task is None or self._reader_task.done():
                        return  # socket is gone -- no further speak_ended is coming
                    await asyncio.sleep(DRAIN_POLL_INTERVAL_S)
        except TimeoutError:
            logger.warning(
                "avatar still speaking after %ss (%s/%s sentences rendered) -- tearing down anyway",
                timeout_s,
                self._speak_turns_ended,
                self._speak_turns_sent,
                extra={"node_name": "heygen_streaming"},
            )

    async def interrupt(self) -> None:
        if self._ws is None:
            raise RuntimeError("LiveAvatarRenderer used outside 'async with' -- no open connection")
        await self._ws.send(json.dumps({"type": "agent.interrupt"}))

    async def _wait_for_state(self, expected_state: str) -> None:
        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "session.stopped":
                raise RuntimeError(f"HeyGen session stopped before reaching state '{expected_state}': {event}")
            if event_type == "session.state_updated" and event.get("state") == expected_state:
                return
        raise RuntimeError(f"Connection closed before session reached state '{expected_state}'")

    async def _read_events_loop(self) -> None:
        """Sole consumer of the HeyGen socket once the session is connected.

        Draining here rather than inside speak_audio() is what keeps sending
        off the critical path (see speak_audio's docstring). It also means
        session.stopped is noticed BETWEEN sentences -- previously nothing
        read the socket except the inline wait, so a session that died while
        we weren't speaking went unnoticed until the next send failed.

        Events are matched on TYPE ONLY, never on the event_id we sent.
        Verified against the real API with scripts/check_heygen_session.py:
        HeyGen mints a FRESH event_id for every event it emits and never
        echoes ours back (its own correlation key is `task.id`). Comparing
        against our event_id is what made the old inline wait never match,
        so every sentence burned the full 30s timeout.
        """
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "agent.speak_started":
                    self._record_first_video()
                elif event_type == "agent.speak_ended":
                    self._speak_turns_ended += 1
                elif event_type == "session.stopped":
                    # Surfaced from speak_audio() instead of raised here --
                    # an exception in a background task has nowhere to go.
                    self._session_error = f"HeyGen session stopped: {event}"
                    logger.error(
                        "heygen session stopped: %s", event, extra={"node_name": "heygen_streaming"}
                    )
                    return
        except websockets.exceptions.ConnectionClosed:
            return

    def _record_first_video(self) -> None:
        # Best-effort latency marker: the turn's first agent.speak_started is
        # when the avatar actually began rendering -- the real "first video"
        # moment the user experiences, not just when we finished sending
        # bytes. Idempotent and non-fatal, mirroring TurnTiming's record_*
        # methods in voice_routes.py.
        if self._ttfv_logged or self._first_speak_at is None:
            return
        self._ttfv_logged = True
        ttfv_ms = round((time.monotonic() - self._first_speak_at) * 1000, 1)
        # logfire captures ttfv_ms as a structured, chartable attribute; the
        # JSON logger only serializes CORRELATION_FIELDS (see logger.py), so
        # the number is put in the message itself to stay visible there too.
        logfire.info("time to first avatar video: {ttfv_ms}ms", ttfv_ms=ttfv_ms)
        logger.info("time to first avatar video: %sms", ttfv_ms, extra={"node_name": "heygen_streaming"})

    async def _keep_alive_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_S)
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "session.keep_alive"}))
