"""Manual check: confirm the planner routes correctly against the real API --
small talk / history-answerable -> CONVERSATIONAL, knowledge questions -> a
rewritten search query. This is the exact failure mode Multimodal_RAG's own
reference planner documented for gpt-4o-mini (misrouting technical
questions to CONVERSATIONAL and skipping retrieval), so this check exists
to catch that regression on our corpus/prompt before it reaches voice_ws.

Usage: python -m scripts.check_planner
"""

import asyncio

from app.services.planner import plan

CASES = [
    ([], "hi there", True),
    ([{"role": "user", "content": "my name is Sam"}], "what's my name?", True),
    ([], "what is OLAP?", False),
    (
        [{"role": "user", "content": "tell me about OLAP"}, {"role": "assistant", "content": "OLAP is ..."}],
        "how is it different from data mining",
        False,
    ),
    ([], "how does a neural network learn?", False),
]


async def main() -> None:
    for history, message, expect_conversational in CASES:
        decision = await plan(history, message)
        verdict = "OK  " if decision.is_conversational == expect_conversational else "FAIL"
        print(f"{verdict} message={message!r} -> is_conversational={decision.is_conversational} query={decision.query!r}")


if __name__ == "__main__":
    asyncio.run(main())
