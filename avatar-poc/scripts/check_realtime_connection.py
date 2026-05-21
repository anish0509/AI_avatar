"""Connectivity check: open an OpenAI Realtime API session, send one text
line, receive streamed audio, and save it as a playable .wav file. Run this
BEFORE building anything else -- proves API key + model + audio format all
work end to end. Manual script, hits the real OpenAI API (not part of the
automated pytest suite).

Usage: python -m scripts.check_realtime_connection
"""

import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000  # OpenAI Realtime API's pcm16 output rate
TEST_TEXT = (
    "I want to understand sales and marketing so that i can create branding for my clients this will help me to increase the revenue of my orgnaization and close more deal" 
    
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tmp" / "connectivity_check.wav"

VERBATIM_INSTRUCTIONS = (
    "You are a text-to-speech engine. Read the user's message aloud, "
    "verbatim, exactly as written. Do not respond, comment, add anything, "
    "or ask questions -- just read the text aloud."
)


async def main() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    url = f"{REALTIME_URL}?model={settings.realtime_model}"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []

    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "model": settings.realtime_model,
                        "output_modalities": ["audio"],
                        "instructions": VERBATIM_INSTRUCTIONS,
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "voice": settings.realtime_voice,
                            }
                        },
                    },
                }
            )
        )

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": TEST_TEXT}],
                    },
                }
            )
        )

        await ws.send(json.dumps({"type": "response.create", "response": {"output_modalities": ["audio"]}}))

        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")
            logger.info(f"event: {event_type}", extra={"node_name": "check_realtime_connection"})

            if event_type == "response.output_audio.delta":
                audio_chunks.append(base64.b64decode(event["delta"]))
            elif event_type == "response.output_audio_transcript.delta":
                transcript_parts.append(event.get("delta", ""))
            elif event_type == "error":
                raise RuntimeError(f"Realtime API error: {event}")
            elif event_type == "response.done":
                break

    if not audio_chunks:
        raise RuntimeError("No audio received -- connectivity check failed")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(audio_chunks))

    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"Saved {OUTPUT_PATH} ({total_bytes} bytes of audio)")
    print(f"Transcript (what the model actually said): {''.join(transcript_parts)}")
    print(f"Text we asked it to speak:                 {TEST_TEXT}")


if __name__ == "__main__":
    asyncio.run(main())
