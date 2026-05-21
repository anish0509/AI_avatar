"""Manual check: drives multiple turns through stream_agent_response() with
the SAME thread_id, proving multi-turn memory actually works -- a follow-up
question using a pronoun ("it", "that") must be resolved by the planner
using the real conversation history, and answered correctly.

Usage: python -m scripts.check_agent
"""

import asyncio

from app.services.agent import stream_agent_response
from app.services.conversation_memory import clear_thread, get_history

THREAD_ID = "check-agent-manual-thread"


async def ask(prompt: str) -> None:
    print(f"You: {prompt}")
    text = ""
    async for delta in stream_agent_response(prompt, THREAD_ID):
        text += delta
    print(f"Agent: {text}\n")


async def main() -> None:
    clear_thread(THREAD_ID)

    await ask("what is OLAP?")
    await ask("how is it different from data mining?")  # pronoun -- needs history to resolve
    await ask("what did I just ask you?")  # conversational -- must skip retrieval
    await ask("what is sales?")  # off-topic -- must abstain, not hallucinate

    print("--- final stored thread history ---")
    for msg in get_history(THREAD_ID):
        print(f"  {msg['role']}: {msg['content'][:100]}")

    clear_thread(THREAD_ID)


if __name__ == "__main__":
    asyncio.run(main())
