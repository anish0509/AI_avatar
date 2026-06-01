"""Transcribes recorded user speech to text via OpenAI's audio
transcription API -- this is the speech-to-text side of voice input, turning
a spoken prompt into the text that flows through the same LLM -> chunker ->
TTS pipeline a typed prompt does. Server-side (not the browser's Web Speech
API) for better Hindi/Hinglish accuracy and to keep the API key off the
client. Verified manually against the real API; not hit by the automated
test suite, same convention as llm_stream.py / the ASRProvider in the
ingestion project for external paid APIs (mocked in tests, verified
manually)."""

from app.core.openai_client import get_openai_client

from app.core.config import settings



async def transcribe_audio(audio: bytes, filename: str = "audio.webm") -> str:
    """Transcribe recorded audio bytes to text. `filename` carries the format
    (extension) so OpenAI knows how to decode the bytes (webm/wav/mp4/...)."""
    result = await get_openai_client().audio.transcriptions.create(
        model=settings.stt_model,
        file=(filename, audio),
    )
    return (result.text or "").strip()
