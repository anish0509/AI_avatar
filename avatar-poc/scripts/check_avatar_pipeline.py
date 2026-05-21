"""Manual end-to-end check: drives the REAL /ws/avatar route (not the service
functions directly) with one or more real questions over a SINGLE connection,
exactly as avatar.html would -- proves the full chain (answer generation ->
chunker -> TTS -> HeyGen LiveAvatar WebSocket) actually works wired together,
not just in isolation, AND that a second question reuses the same HeyGen
session and TTS socket instead of rebuilding both. Can't show you the
rendered video itself (that's LiveKit, browser-only -- see avatar.html for
that), but confirms every event fires in the right order and timestamps each
one. Mirrors scripts/check_voice_pipeline.py's verification style for
/ws/voice.

Pass multiple questions to prove reuse across turns: a "session" frame (and a
HeyGen session_id) should appear only ONCE, before the first question's
answer, and NOT again before the second's -- if reuse is working, the second
question's first "text" event should land almost immediately after its
"meta", with no session-setup delay in between.

Watch the timestamps on "session" versus "meta" for the FIRST question: HeyGen
setup and answer generation run concurrently, so whichever finishes first
prints first. If "session" consistently lands well after the first "text",
HeyGen setup is the bottleneck for time-to-first-sound, not the LLM.

Requires the server to already be running (see Usage) AND HEYGEN_API_KEY set
in .env. Hits real paid APIs (OpenAI + HeyGen, + Pinecone if
ANSWER_SOURCE=rag/agent) -- not part of the automated test suite.

Reuse across turns needs HEYGEN_IS_SANDBOX=false: sandbox sessions are capped
at 60s from creation, so asking a second question after that point will hit a
dead HeyGen session regardless of anything this script or the backend does --
that's a real sandbox limit, not a bug to chase.

Usage:
    # in one terminal:
    uvicorn app.main:app
    # in another -- one question, or several to prove reuse:
    python -m scripts.check_avatar_pipeline "your question here"
    python -m scripts.check_avatar_pipeline "What is RAG?" "What is a neural network?"
"""

import asyncio
import json
import os
import sys
import time

import websockets

# AVATAR_PORT lets this run against a fresh server on a spare port without
# disturbing one already on 8000.
PORT = os.getenv("AVATAR_PORT", "8000")
WS_URL = f"ws://127.0.0.1:{PORT}/ws/avatar"
DEFAULT_QUESTIONS = ["What is the difference between OLAP and data mining?"]


async def _ask_one_turn(ws: websockets.ClientConnection, question: str, start: float) -> tuple[list[str], int, int]:
    def elapsed() -> str:
        return f"{round(time.monotonic() - start, 2):>6}s"

    print(f'\n{elapsed()}  >>> asking: "{question}"')
    await ws.send(json.dumps({"question": question}))

    text_parts: list[str] = []
    speaking_frames = 0
    session_frames = 0
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type")

        if event_type == "session":
            session_frames += 1
            # A NEW HeyGen session. On a second-or-later question this means
            # reuse did NOT happen -- either this is genuinely the first
            # question, or an idle release / error rebuilt the session since.
            print(
                f"{elapsed()}  session  : {event['session_id']} "
                f"(max_duration={event.get('max_session_duration')}s) -- NEW HeyGen session built here"
            )
            # A real browser joins LiveKit here. This script can't render
            # video, so it reports ready immediately -- the backend holds
            # the first sentence until it sees this.
            await ws.send(json.dumps({"type": "ready"}))
            print(f"{elapsed()}  ready    : sent (a browser would have joined the LiveKit room first)")
        elif event_type == "meta":
            # Proof of which answer path (agent/rag/gpt) actually ran, same
            # as check_voice_pipeline.py's META line.
            print(f"{elapsed()}  META     : {json.dumps({k: v for k, v in event.items() if k != 'type'})}")
        elif event_type == "text":
            text_parts.append(event["text"])
            print(f"{elapsed()}  text     : {event['text']}")
        elif event_type == "speaking":
            speaking_frames += 1
            print(f"{elapsed()}  speaking : (a sentence's TTS audio was pushed to HeyGen)")
        elif event_type == "error":
            raise RuntimeError(f"/ws/avatar error: {event.get('detail')}")
        elif event_type == "done":
            break

    if not text_parts:
        raise RuntimeError("No answer text received for this question -- pipeline check failed")
    return text_parts, speaking_frames, session_frames


async def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    print(f"Connecting {WS_URL} ...")
    print(f"Asking {len(questions)} question(s) over ONE connection to check session/socket reuse.")

    start = time.monotonic()
    session_frame_count = 0
    all_answers: list[str] = []
    total_speaking_frames = 0

    async with websockets.connect(WS_URL) as ws:
        for question in questions:
            text_parts, speaking_frames, session_frames = await _ask_one_turn(ws, question, start)
            all_answers.append(" ".join(text_parts))
            total_speaking_frames += speaking_frames
            session_frame_count += session_frames

    elapsed_s = round(time.monotonic() - start, 2)

    print("\n--- answers ---")
    for i, answer in enumerate(all_answers, start=1):
        print(f"{i}. {answer}")

    print(f"\n{total_speaking_frames} sentence(s) across {len(questions)} question(s); {elapsed_s}s end to end.")
    if len(questions) > 1:
        if session_frame_count > 1:
            print(f"NOTE: saw {session_frame_count} distinct 'session' frames printed above -- check whether that's")
            print("expected (idle release, or an error forced a rebuild) before trusting the timing numbers.")
        print("Reuse check: exactly one 'NEW HeyGen session built here' line above means the later question(s)")
        print("reused it for free. More than one means a rebuild happened -- see the notes printed at each one.")
    print('Server logs add "time to first avatar video: <N>ms" -- when HeyGen began rendering. The true')
    print("time-to-first-SOUND is browser-only; open avatar.html, which measures it from the audio track.")


if __name__ == "__main__":
    asyncio.run(main())
