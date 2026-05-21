"""Unit tests for TurnTiming (app/api/voice_routes.py) -- the server-side
time-to-first-response measurement (see problem.md P0.1). Pure logic: the
clock is patched, so no real timing/flakiness, and the record_* methods
return the measured milliseconds, so behaviour is asserted directly rather
than by scraping log output. The full pipeline integration (that _consume
actually calls these on the first text/audio) is already covered end-to-end
by tests/test_orchestration.py, which drives voice_ws.
"""

from app.api import voice_routes
from app.api.voice_routes import TurnTiming


def test_measures_first_text_and_first_audio_elapsed(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(voice_routes.time, "monotonic", lambda: clock["now"])

    timing = TurnTiming(start=100.0, log_extra={"node_name": "voice_ws"})

    clock["now"] = 100.4  # 400ms after the prompt was received
    assert timing.record_first_text() == 400.0

    clock["now"] = 100.7  # 700ms after start -> TTS added ~300ms on top of first text
    assert timing.record_first_audio() == 700.0


def test_only_the_first_text_and_first_audio_are_timed(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(voice_routes.time, "monotonic", lambda: clock["now"])

    timing = TurnTiming(start=0.0, log_extra={})

    clock["now"] = 0.5
    assert timing.record_first_text() == 500.0
    clock["now"] = 0.9
    assert timing.record_first_text() is None  # subsequent sentences must not re-time

    clock["now"] = 1.2
    assert timing.record_first_audio() == 1200.0
    clock["now"] = 2.0
    assert timing.record_first_audio() is None  # subsequent audio bytes must not re-time
