"""Plan each turn, retrieve supporting content, and stream an answer with conversation history."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import logfire
from langsmith import traceable
from app.core.openai_client import get_openai_client

from app.core.config import settings
from app.core.logger import get_logger
from app.services.conversation_memory import Message, append_message, get_history
from app.services.planner import PlannerDecision, plan
from app.services.rag_stream import RetrievalContext, retrieve_context, stream_grounded_answer

logger = get_logger(__name__)


CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are a friendly assistant. Answer the user's latest message using "
    "the CONVERSATION HISTORY below."
)


@dataclass
class AgentDecision:
    planner: PlannerDecision
    retrieval: RetrievalContext | None  # None when conversational -- no retrieval ran


def _format_history(history: list[Message]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


@traceable(name="plan_turn", run_type="chain")
async def plan_turn(prompt: str, thread_id: str) -> AgentDecision:
    """Run the planner (and retrieval, if the planner says this turn needs
    a lookup) for `prompt` against `thread_id`'s history. Logs the decision
    once here so it's recorded exactly once per turn regardless of caller."""
    with logfire.span("Agent: plan turn", thread_id=thread_id):
        history = await get_history(thread_id)
        decision = await plan(history, prompt)

        logger.info(
            "planner decision",
            extra={
                "node_name": "agent",
                "thread_id": thread_id,
                "is_conversational": decision.is_conversational,
                "planner_query": decision.query,
            },
        )

        if decision.is_conversational:
            return AgentDecision(planner=decision, retrieval=None)

        retrieval = await retrieve_context(decision.query)
        return AgentDecision(planner=decision, retrieval=retrieval)


@traceable(name="conversational_answer", run_type="llm")
async def _stream_conversational_answer(history: list[Message], message: str) -> AsyncIterator[str]:
    with logfire.span("Conversational answer (no retrieval)"):
        user_prompt = f"CONVERSATION HISTORY:\n{_format_history(history)}\n\nLATEST MESSAGE:\n{message}"
        stream = await get_openai_client().chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# NOT @traceable, deliberately -- see the "Why no @traceable" note below.
async def stream_agent_answer(prompt: str, thread_id: str, decision: AgentDecision) -> AsyncIterator[str]:
    """Stream the answer for an already-planned decision, persisting the turn
    to thread memory as it goes so the next turn can see it.

    Ordering here is load-bearing:

    1. Read history FIRST. The prompt is appended below, and if that happened
       before the read, this turn's own question would appear twice -- once in
       the history block and again as the question being asked.
    2. Append the user message BEFORE generating. Both appends used to happen
       after the stream finished, so a Stop or disconnect mid-answer lost the
       entire turn and the next question had no idea this one was ever asked
       (P3 in architecture-and-query-flow.md).
    3. Append whatever the assistant actually produced in a finally, including
       a partial answer when the user stopped mid-sentence -- they heard that
       much, so the history should say so rather than pretending silence.

    Why no @traceable on this one function (every sibling has it): LangSmith's
    decorator wraps an async generator in its own generator that does NOT
    propagate aclose() inward. With it applied, the finally below did not run
    when the consumer stopped early -- it ran whenever GC eventually collected
    the abandoned generator, i.e. possibly after the next question had already
    read the history. Measured, not assumed: with the decorator the
    stopped-mid-stream test failed with the write landing after the assertion;
    without it, the same test passes. The logfire span below still covers this
    function, so the observability loss is one duplicate LangSmith span, which
    is a cheap price for a deterministic write. `_produce()`'s
    contextlib.aclosing() is what actually triggers the close -- the two go
    together and removing either one silently reintroduces the bug.
    """
    with logfire.span(
        "Agent: stream answer", thread_id=thread_id, is_conversational=decision.planner.is_conversational
    ):
        history = await get_history(thread_id)
        await append_message(thread_id, "user", prompt)

        answer_parts: list[str] = []
        try:
            if decision.retrieval is None:
                async for delta in _stream_conversational_answer(history, prompt):
                    answer_parts.append(delta)
                    yield delta
            else:
                async for delta in stream_grounded_answer(prompt, decision.retrieval, history):
                    answer_parts.append(delta)
                    yield delta
        finally:
            # Only reached promptly because (a) this function is not
            # @traceable and (b) _produce() wraps the stream in
            # contextlib.aclosing(). See the docstring -- both are required.
            if answer_parts:
                await append_message(thread_id, "assistant", "".join(answer_parts))


@traceable(name="stream_agent_response", run_type="chain")
async def stream_agent_response(prompt: str, thread_id: str) -> AsyncIterator[str]:
    """One-call entry point (plan + stream) for the check scripts.
    voice_routes.py uses plan_turn + stream_agent_answer directly so it can
    report the decision (meta event) before any text streams."""
    decision = await plan_turn(prompt, thread_id)
    async for delta in stream_agent_answer(prompt, thread_id, decision):
        yield delta
