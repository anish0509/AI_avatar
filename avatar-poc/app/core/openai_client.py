"""Create the shared API client only when a request needs it."""

import logfire
from openai import AsyncOpenAI

from app.core.config import settings

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
        logfire.instrument_openai(_client)
    return _client
