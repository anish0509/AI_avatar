"""Stream text completions from OpenAI."""

from collections.abc import AsyncIterator

import logfire
from langsmith import traceable
from app.core.openai_client import get_openai_client

from app.core.config import settings



@traceable(name="stream_llm_response", run_type="llm")
async def stream_llm_response(prompt: str) -> AsyncIterator[str]:
    """Yield text pieces (deltas) as the LLM generates its response to `prompt`."""
    with logfire.span("LLM response (no retrieval)", prompt=prompt[:200]):
        stream = await get_openai_client().chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
