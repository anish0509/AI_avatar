"""Streams text from OpenAI's chat completions API, piece by piece -- this
is the source of the 'streaming text' that text_chunker.py buffers into
sentences. Verified manually against the real API
(scripts/check_llm_stream.py); not hit by the automated test suite, same
convention as the ASRProvider in the ingestion project for external paid
APIs (mocked in tests, verified manually).

Observability (Step 8): logfire.instrument_openai() auto-traces every raw
API call through this client (request/response, tokens, latency) --
configured once, at client construction. @traceable adds a named,
higher-level LangSmith span for the whole function call. Both no-op
without their respective tokens configured (see app/core/observability.py).
"""

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
