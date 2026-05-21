"""Manual end-to-end check: drives the REAL /ws/voice route (not the
service functions directly) with a real question, exactly as the browser
UI would -- proves retrieval -> grounded LLM -> TextChunker -> TTS -> audio
bytes all actually work wired together, not just in isolation. Saves the
received audio as a playable .wav so the full loop can be sanity-checked
by ear, mirroring scripts/check_realtime_connection.py's verification style.

Requires the server to already be running (see Usage). Hits real paid APIs
(OpenAI + Pinecone) -- not part of the automated test suite.

Usage:
    # in one terminal:
    ANSWER_SOURCE=rag uvicorn app.main:app
    # in another:
    python -m scripts.check_voice_pipeline "your question here"
"""

import asyncio
import json
import os
import sys
import wave
from pathlib import Path

import websockets

# VOICE_PORT lets this run against a fresh server on a spare port without
# disturbing one already on 8000.
WS_URL = f"ws://127.0.0.1:{os.getenv('VOICE_PORT', '8000')}/ws/voice"
SAMPLE_RATE = 24000  # matches RealtimeApiSpeaker's output rate
DEFAULT_QUESTION = "What is the difference between OLAP and data mining?"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tmp" / "voice_pipeline_check.wav"


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f'Question: "{question}"')
    print(f"Connecting to {WS_URL} ...")

    text_parts: list[str] = []
    audio_chunks: list[bytes] = []

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"prompt": question}))

        async for raw in ws:
            if isinstance(raw, bytes):
                audio_chunks.append(raw)
                continue

            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "meta":
                # The per-turn answer-source announcement -- this is the proof
                # of which path (RAG vs direct LLM) actually ran.
                print(f"  META : {json.dumps({k: v for k, v in event.items() if k != 'type'})}")
            elif event_type == "text":
                text_parts.append(event["text"])
                print(f"  text : {event['text']}")
            elif event_type == "error":
                raise RuntimeError(f"/ws/voice error: {event.get('detail')}")
            elif event_type == "done":
                break

    if not audio_chunks:
        raise RuntimeError("No audio received -- pipeline check failed")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(audio_chunks))

    total_bytes = sum(len(c) for c in audio_chunks)
    print("\n--- full spoken answer (text) ---")
    print(" ".join(text_parts))
    print(f"\nSaved {OUTPUT_PATH} ({total_bytes} bytes of audio, {len(audio_chunks)} chunks)")


if __name__ == "__main__":
    asyncio.run(main())
