"""Times the sequential steps of a setup sequence and reports them as one log
line plus structured logfire attributes.

Extracted rather than repeated because three separate services each open an
external WebSocket whose setup cost was invisible -- `heygen_streaming`,
`realtime_tts` and `realtime_stt`. Two of those now run concurrently with other
work, and a concurrent stage leaves no gap in a trace to infer its duration
from, so it has to be measured at the source or not at all (see P17 in
architecture-and-query-flow.md, where inferring it produced a wrong number).

Same reasoning as `app/core/task_bridge.py`: shared setup/teardown logic lives
in `app/core/` so a second caller reuses it instead of growing a parallel
implementation.
"""

import contextlib
import time
from collections.abc import Iterator

import logfire

from app.core.logger import get_logger

logger = get_logger(__name__)


class StepTimer:
    """Accumulates per-step durations. `mark()` closes the step that just ran,
    so steps are recorded in the order they complete and each duration is the
    time since the previous mark."""

    def __init__(self, label: str, node_name: str) -> None:
        self._label = label
        self._node_name = node_name
        self._started = time.monotonic()
        self._last = self._started
        self.steps: dict[str, float] = {}

    def mark(self, step: str) -> float:
        now = time.monotonic()
        elapsed_ms = round((now - self._last) * 1000, 1)
        self.steps[f"{step}_ms"] = elapsed_ms
        self._last = now
        return elapsed_ms

    def total_ms(self) -> float:
        return round((time.monotonic() - self._started) * 1000, 1)

    def log(self, **log_fields: object) -> float:
        """Emit the breakdown and return the total, so callers can assert on it
        without having to read log output."""
        total = self.total_ms()
        # logfire gets each step as its own structured attribute so they can be
        # charted and compared across runs; the JSON logger only serializes
        # CORRELATION_FIELDS (see logger.py), so the numbers go in its message.
        logfire.info(self._label + ": {total_ms}ms", total_ms=total, **self.steps)
        breakdown = ", ".join(f"{name.removesuffix('_ms')} {value}ms" for name, value in self.steps.items())
        logger.info(
            "%s: %sms total (%s)",
            self._label,
            total,
            breakdown or "no steps marked",
            extra={"node_name": self._node_name, **log_fields},
        )
        return total


@contextlib.contextmanager
def step_timer(label: str, node_name: str) -> Iterator[StepTimer]:
    """Wraps the timed work in a logfire span AND yields a StepTimer for the
    per-step breakdown. The span is what makes the stage visible as its own
    bar in a trace; the steps are what make it actionable."""
    with logfire.span(label):
        yield StepTimer(label, node_name)
