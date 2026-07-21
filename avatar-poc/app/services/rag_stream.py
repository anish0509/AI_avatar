"""Retrieve document context and stream grounded answers. Abstain below the similarity threshold."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import logfire
from langsmith import traceable
from app.core.openai_client import get_openai_client

from app.core.config import settings
from app.core.logger import get_logger
from app.services.conversation_memory import Message
from app.services.retrieval import RetrievedChunk, search

logger = get_logger(__name__)


ABSTAIN_MESSAGE = "I don't have enough information in the knowledge base to answer that."

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the CONTEXT below. If the context does not contain enough "
    "information to answer, say so clearly instead of guessing."
)


@dataclass
class RetrievalContext:
    """The result of the retrieval + grounding decision for one query --
    everything needed both to generate the answer and to report (to the
    browser and logs) exactly what retrieval did."""

    chunks: list[RetrievedChunk]
    top_score: float
    grounded: bool

    @property
    def sources(self) -> list[str]:
        """Unique source files among the retrieved chunks, order-preserving."""
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen[chunk.source_file] = None
        return list(seen)


def is_grounded(chunks: list[RetrievedChunk]) -> bool:
    """True if the top retrieved chunk's score clears the relevance floor --
    i.e. there's real supporting context for this query, not just whatever
    dense search's least-bad match happened to be."""
    if not chunks:
        return False
    return chunks[0].score >= settings.retrieval_score_floor


@traceable(name="retrieve_context", run_type="chain")
async def retrieve_context(prompt: str) -> RetrievalContext:
    """Run dense retrieval + the grounding gate for `prompt`, log the
    decision, and return it. Logged here (rather than inside the streaming
    call) so the decision is recorded exactly once per query no matter which
    caller drives it (voice_routes or the check scripts).

    Async only because search() is (it runs Pinecone's blocking SDK on a
    thread so the event loop stays free) -- the gate logic itself is pure."""
    with logfire.span("Retrieval + grounding gate", query=prompt[:200]):
        chunks = await search(prompt)
        top_score = chunks[0].score if chunks else 0.0
        grounded = is_grounded(chunks)

        context = RetrievalContext(chunks=chunks, top_score=top_score, grounded=grounded)

        logfire.info(
            "gate {decision}: top_score={top_score} floor={floor}",
            decision="ALLOWED" if grounded else "BLOCKED",
            top_score=round(top_score, 4),
            floor=settings.retrieval_score_floor,
            sources=context.sources,
        )
        logger.info(
            "retrieval gate check",
            extra={
                "node_name": "rag_stream",
                "query": prompt[:80],
                "top_score": round(top_score, 4),
                "score_floor": settings.retrieval_score_floor,
                "gate_decision": "ALLOWED" if grounded else "BLOCKED",
                "retrieved_sources": context.sources,
            },
        )
        return context


def _format_history(history: list[Message]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk], history: list[Message] | None = None) -> str:
    """Pure function: assemble retrieved chunks (+ optional conversation
    history, for the agent's multi-turn use -- absent for the plain
    single-turn RAG path) + the question into the user-message text sent to
    the LLM. Only called once grounding has already passed, so chunks is
    never empty here in practice -- the empty case is still handled
    defensively."""
    history_block = f"\n\nCONVERSATION HISTORY:\n{_format_history(history)}" if history else ""

    if not chunks:
        return f"CONTEXT:\n(No relevant context was found for this question.){history_block}\n\nQUESTION:\n{question}"

    context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
    return f"CONTEXT:\n{context}{history_block}\n\nQUESTION:\n{question}"


@traceable(name="stream_grounded_answer", run_type="chain")
async def stream_grounded_answer(
    prompt: str, context: RetrievalContext, history: list[Message] | None = None
) -> AsyncIterator[str]:
    """Stream the answer for an already-retrieved context. If the context
    isn't grounded, abstain WITHOUT calling the LLM (a single fixed
    message); otherwise stream a grounded OpenAI completion. `history` is
    optional so the plain single-turn RAG path (stream_rag_response) is
    unaffected -- only the agent (Step 5) passes it."""
    if not context.grounded:
        with logfire.span("Grounding gate BLOCKED -- abstaining, no LLM call"):
            yield ABSTAIN_MESSAGE
        return

    with logfire.span("Grounded LLM synthesis", chunk_count=len(context.chunks)):
        grounded_prompt = build_grounded_prompt(prompt, context.chunks, history)

        stream = await get_openai_client().chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": grounded_prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


@traceable(name="stream_rag_response", run_type="chain")
async def stream_rag_response(prompt: str) -> AsyncIterator[str]:
    """One-call entry point (retrieve + gate + stream) with the same
    `str -> AsyncIterator[str]` contract as stream_llm_response. Used by the
    check scripts; voice_routes uses retrieve_context + stream_grounded_answer
    directly so it can report the retrieval decision up front."""
    context = await retrieve_context(prompt)
    async for delta in stream_grounded_answer(prompt, context):
        yield delta
