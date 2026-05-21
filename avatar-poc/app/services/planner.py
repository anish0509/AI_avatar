"""Routes each agent turn: CONVERSATIONAL (small talk, or answerable from
history alone -- skip retrieval) or a rewritten, self-contained search
query (needs a knowledge-base lookup). Pattern adapted from Multimodal_RAG's
planner node, rewritten domain-agnostic for this project's general-document
corpus (their reference was hard-coded to Kubernetes/Intel), and for lean
async instead of LangGraph -- a plain coroutine, not a graph node.

build_planner_prompt() and parse_planner_output() are pure and unit tested.
The actual LLM call is verified manually against the real API
(scripts/check_planner.py), same convention every other paid-API service in
this codebase follows.

Observability (Step 8): the planner's routing decision is exactly the kind
of thing worth tracing -- confirmed (Multimodal_RAG's own docs) that a
misrouted planner (skipping retrieval when it shouldn't) is a real,
observed failure mode, not hypothetical, so being able to see the routing
decision per query in the LangSmith/Logfire dashboards matters here more
than in most other functions.
"""

from dataclasses import dataclass

import logfire
from langsmith import traceable
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.conversation_memory import Message

_client = AsyncOpenAI(api_key=settings.openai_api_key)
logfire.instrument_openai(_client)

CONVERSATIONAL = "CONVERSATIONAL"

PLANNER_PROMPT_TEMPLATE = """You are the ROUTING PLANNER for a knowledge-base assistant.
Your ONLY job is to classify the user's LATEST message. You must NOT answer it.

Produce EXACTLY ONE of these two outputs:

1. The single word: CONVERSATIONAL
   Use this when the latest message is small talk (greeting, thanks, "who are
   you") OR is fully answerable from the CONVERSATION HISTORY alone
   (e.g. "what did I just ask", "what's my name").

2. A SEARCH QUERY (a single line of plain text)
   Use this when answering needs a knowledge-base lookup.
   - Make the query SELF-CONTAINED: resolve pronouns/references using the history.
   - Keep it concise and keyword-focused for semantic retrieval.

OUTPUT CONTRACT (STRICT):
- Output ONLY the word CONVERSATIONAL, or ONLY the search query text.
- No quotes, no backticks, no markdown, no explanation, no leading label. Exactly one line.

EXAMPLES:
History: (none)                             | Latest: "hi there"                -> CONVERSATIONAL
History: "user: my name is Sam"             | Latest: "what's my name?"         -> CONVERSATIONAL
History: (none)                             | Latest: "what is OLAP?"           -> what is OLAP
History: "user: tell me about OLAP"         | Latest: "how is it different from data mining" -> how is OLAP different from data mining

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{message}"

Your output:"""


@dataclass
class PlannerDecision:
    is_conversational: bool
    query: str  # the rewritten, self-contained search query; the sentinel word if conversational


def _format_history(history: list[Message]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def build_planner_prompt(history: list[Message], message: str) -> str:
    return PLANNER_PROMPT_TEMPLATE.format(history=_format_history(history), message=message)


def parse_planner_output(raw: str) -> PlannerDecision:
    """Robust parse: strip wrapping quotes/backticks/trailing punctuation so
    a lightly-decorated 'CONVERSATIONAL' still routes correctly. Anything
    that doesn't reduce to the sentinel is treated as a search query."""
    decision = raw.strip().strip("`\"'").strip()
    is_conversational = decision.upper().rstrip(".!").strip() == CONVERSATIONAL
    if is_conversational:
        return PlannerDecision(is_conversational=True, query=CONVERSATIONAL)
    return PlannerDecision(is_conversational=False, query=decision)


@traceable(name="plan", run_type="llm")
async def plan(history: list[Message], message: str) -> PlannerDecision:
    with logfire.span("Planner decision", message=message[:200]):
        prompt = build_planner_prompt(history, message)
        response = await _client.chat.completions.create(
            model=settings.planner_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        decision = parse_planner_output(raw)
        logfire.info(
            "routed to: {route}",
            route=CONVERSATIONAL if decision.is_conversational else decision.query,
            is_conversational=decision.is_conversational,
        )
        return decision
