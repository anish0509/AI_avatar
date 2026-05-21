"""Manual check: use the REAL RealtimeApiTranscriber (not one-off script
code) to stream known-text audio -- generated via the REAL
RealtimeApiSpeaker, same verbatim-TTS trick as check_speaker.py -- in small
chunks, and confirm streaming deltas arrive incrementally and the final
transcript matches what was spoken.

Usage: python -m scripts.check_realtime_stt
"""

import asyncio

from app.services.realtime_stt import RealtimeApiTranscriber
from app.services.realtime_tts import SAMPLE_RATE, RealtimeApiSpeaker

TEXT = "This is a streaming transcription test."
CHUNK_SIZE = 4800  # ~100ms @ 24kHz/16-bit mono
CHUNK_INTERVAL_SECONDS = 0.05


async def generate_ground_truth_audio(text: str) -> bytes:
    async with RealtimeApiSpeaker() as speaker:
        chunks = [chunk async for chunk in speaker.speak(text)]
    return b"".join(chunks)


async def main() -> None:
    print(f"Generating ground-truth audio for: {TEXT!r}")
    speech = await generate_ground_truth_audio(TEXT)
    # Append ~1.2s of real silence: TTS output ends abruptly with no trailing
    # quiet, but server VAD only finalizes a segment after it hears
    # silence_duration_ms of actual silence -- exactly what a real mic streams
    # after you stop talking. Without this, raising the VAD gap would leave the
    # last utterance uncommitted (it's the audio, not the model, that ends it).
    pcm = speech + b"\x00\x00" * int(SAMPLE_RATE * 1.2)
    duration = len(pcm) / 2 / SAMPLE_RATE
    print(f"  -> {len(speech)} bytes speech + trailing silence = {duration:.2f}s")

    async with RealtimeApiTranscriber() as transcriber:

        async def feed_audio() -> None:
            for i in range(0, len(pcm), CHUNK_SIZE):
                await transcriber.send_audio(pcm[i : i + CHUNK_SIZE])
                await asyncio.sleep(CHUNK_INTERVAL_SECONDS)
            await asyncio.sleep(2)  # let server-side VAD notice end of speech

        feeder = asyncio.create_task(feed_audio())
        final_transcript = ""
        try:
            async with asyncio.timeout(15):
                async for event in transcriber.events():
                    if event.kind == "delta":
                        print(f"  delta -> {event.text!r}")
                    else:
                        print(f"  FINAL -> {event.text!r}")
                        final_transcript = event.text
        except TimeoutError:
            pass
        finally:
            feeder.cancel()

    print(f"\nExpected: {TEXT!r}")
    print(f"Got:      {final_transcript!r}")
    print("MATCH" if final_transcript.strip().lower() == TEXT.strip().lower() else "MISMATCH (see above)")


if __name__ == "__main__":
    asyncio.run(main())
