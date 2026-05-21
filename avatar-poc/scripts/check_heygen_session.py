"""Connectivity check: mint a HeyGen LiveAvatar LITE session, connect its
WebSocket, and dump every raw server event type for ~20s while sending 1s of
silent PCM16 audio. Run this BEFORE trusting app/services/heygen_streaming.py
-- proves the API key, avatar_id, and (most importantly) the assumptions that
code depends on: that a LITE session actually returns a ws_url, that the
WebSocket needs no auth header, and the real event-name spellings for
"session.state_updated"/"agent.speak_started". Deliberately uses raw
httpx/websockets calls rather than LiveAvatarRenderer, so it verifies those
assumptions independently instead of trusting the code built on top of them
(see bug-report-streaming-stt.md Bug 1 for why: code once waited on an event
name that didn't exist and hung forever -- caught only by a throwaway script
dumping raw events against the real API before dependent code was written).

Usage: python -m scripts.check_heygen_session
"""

import asyncio
import base64
import json

import httpx
import websockets

from app.core.config import settings
from app.core.logger import get_logger
from app.services.heygen_streaming import HEYGEN_SAMPLE_RATE, _raise_for_status_with_body, build_token_request

logger = get_logger(__name__)

DUMP_DURATION_S = 20.0
SILENT_AUDIO_SECONDS = 1.0


async def main() -> None:
    if not settings.heygen_api_key:
        raise RuntimeError("HEYGEN_API_KEY is not set in .env")

    async with httpx.AsyncClient(base_url=settings.heygen_api_base, timeout=30.0) as http:
        print(f"1) POST /sessions/token  (avatar_id={settings.heygen_avatar_id}, sandbox={settings.heygen_is_sandbox})")
        token_resp = await http.post(
            "/sessions/token",
            headers={"X-API-KEY": settings.heygen_api_key},
            json=build_token_request(
                settings.heygen_avatar_id,
                settings.heygen_is_sandbox,
                settings.heygen_video_quality,
                settings.heygen_video_encoding,
            ),
        )
        _raise_for_status_with_body(token_resp)
        token_body = token_resp.json()
        print(f"   response: {token_body}")
        session_token = token_body["data"]["session_token"]

        print("2) POST /sessions/start")
        start_resp = await http.post("/sessions/start", headers={"Authorization": f"Bearer {session_token}"})
        _raise_for_status_with_body(start_resp)
        data = start_resp.json()["data"]
        print(f"   response data: {data}")

        ws_url = data.get("ws_url")
        session_id = data["session_id"]
        if not ws_url:
            print("\n!!! NO ws_url IN RESPONSE -- LITE backend-driven audio push is NOT available.")
            print("!!! Stop here and see the Fallback section of the integration plan.")
            await http.post(
                "/sessions/stop",
                headers={"X-API-KEY": settings.heygen_api_key},
                json={"session_id": session_id, "reason": "USER_CLOSED"},
            )
            return

        print(f"   ws_url present: {ws_url[:60]}...")
        print(f"\n3) Connecting WebSocket (no auth header) and dumping events for {DUMP_DURATION_S}s...")

        connected = False
        try:
            async with websockets.connect(ws_url) as ws:

                async def dump_events() -> None:
                    nonlocal connected
                    async for raw in ws:
                        event = json.loads(raw)
                        print(f"   <- {event}")
                        if event.get("type") == "session.state_updated" and event.get("state") == "connected":
                            connected = True

                dumper = asyncio.create_task(dump_events())

                # Wait for "connected" before sending anything.
                for _ in range(100):
                    if connected:
                        break
                    await asyncio.sleep(0.1)
                if not connected:
                    print("   !!! never saw session.state_updated/connected within 10s")

                print(f"\n4) Sending {SILENT_AUDIO_SECONDS}s of silent PCM16 audio via agent.speak...")
                silent_pcm = b"\x00\x00" * int(HEYGEN_SAMPLE_RATE * SILENT_AUDIO_SECONDS)
                event_id = "check-heygen-session-probe"
                await ws.send(
                    json.dumps(
                        {"type": "agent.speak", "audio": base64.b64encode(silent_pcm).decode("ascii"), "event_id": event_id}
                    )
                )
                await ws.send(json.dumps({"type": "agent.speak_end", "event_id": event_id}))

                try:
                    await asyncio.wait_for(dumper, timeout=DUMP_DURATION_S)
                except TimeoutError:
                    dumper.cancel()
        finally:
            print("\n5) POST /sessions/stop")
            stop_resp = await http.post(
                "/sessions/stop",
                headers={"X-API-KEY": settings.heygen_api_key},
                json={"session_id": session_id, "reason": "USER_CLOSED"},
            )
            print(f"   response: {stop_resp.json()}")

    print("\nDone. Check above: did 'session.state_updated'/connected and 'agent.speak_started' actually appear?")


if __name__ == "__main__":
    asyncio.run(main())
