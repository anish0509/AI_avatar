"""Manual check: use the REAL RealtimeApiSpeaker (not one-off script code)
to speak several sentences sequentially on ONE connection, proving
connection reuse actually works against the real API -- not just in the
mocked unit tests. Saves the combined audio as a single .wav.

Usage: python -m scripts.check_speaker
"""

import asyncio
import wave
from pathlib import Path

from app.services.realtime_tts import SAMPLE_RATE, RealtimeApiSpeaker

SENTENCES = [
    "Welcome to the sales marathon program.",
    "Today we will focus on objection handling.",
    "Let's begin with the first technique: acknowledge, then reframe.",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tmp" / "check_speaker.wav"


async def main() -> None:
    audio_chunks: list[bytes] = []

    async with RealtimeApiSpeaker() as speaker:
        for sentence in SENTENCES:
            print(f"Speaking: {sentence}")
            chunk_count = 0
            async for chunk in speaker.speak(sentence):
                audio_chunks.append(chunk)
                chunk_count += 1
            print(f"  -> received {chunk_count} audio chunks")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(audio_chunks))

    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"\nSaved {OUTPUT_PATH} ({total_bytes} bytes, ONE connection, {len(SENTENCES)} sentences)")


if __name__ == "__main__":
    asyncio.run(main())
