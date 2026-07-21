"""Transcribe recorded audio with OpenAI."""

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
