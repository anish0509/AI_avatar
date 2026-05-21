from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.transcription_routes import transcribe


class FakeRequest:
    """Duck-typed stand-in for Starlette's Request: exposes the raw body and
    a headers dict, which is all the transcribe route reads."""

    def __init__(self, body: bytes, content_type: str = "audio/webm") -> None:
        self._body = body
        self.headers = {"content-type": content_type}

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    with patch("app.api.transcription_routes.transcribe_audio", AsyncMock(return_value="hello world")) as mock_stt:
        result = await transcribe(FakeRequest(b"fake-audio-bytes"))

    assert result == {"text": "hello world"}
    mock_stt.assert_awaited_once()
    # filename derived from the webm content-type
    assert mock_stt.await_args.kwargs["filename"] == "audio.webm"


@pytest.mark.asyncio
async def test_transcribe_empty_body_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        await transcribe(FakeRequest(b""))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_transcribe_openai_failure_raises_502():
    with patch("app.api.transcription_routes.transcribe_audio", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(HTTPException) as exc_info:
            await transcribe(FakeRequest(b"fake-audio-bytes"))

    assert exc_info.value.status_code == 502
