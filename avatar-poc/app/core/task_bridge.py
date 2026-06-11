"""Shared asyncio task-bridging helper, extracted from voice_routes.py so
the new /ws/transcribe route can reuse the same cancellation-safety logic
instead of duplicating it (see Day 6 in day-wise-implementation.md for the
bug this originally fixed)."""

import asyncio


async def wait_and_cancel_rest(tasks: tuple[asyncio.Task, ...], return_when: str) -> None:
    """Runs `tasks` concurrently and, once `return_when` is satisfied,
    guarantees every task is either awaited to completion or cancelled --
    even if this coroutine itself gets cancelled from the outside (e.g. an
    outer watcher task wins a race), so a child task can never be silently
    orphaned still running, or left holding an unretrieved exception.
    Re-raises the first real exception found among the tasks, if any."""
    try:
        await asyncio.wait(tasks, return_when=return_when)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    first_exc: BaseException | None = None
    for task in tasks:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None and first_exc is None:
            first_exc = exc
    if first_exc is not None:
        raise first_exc
