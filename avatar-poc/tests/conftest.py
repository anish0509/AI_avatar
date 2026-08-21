"""Configure local tracing and isolate conversation stores between tests."""

import logfire
import pytest


def pytest_configure(config) -> None:
    logfire.configure(token=None, send_to_logfire=False)


@pytest.fixture(autouse=True)
def _pin_in_memory_conversation_store(monkeypatch):
    """Reset the store per test to avoid shared state and cross-event-loop Redis clients."""
    from app.services import conversation_memory as mem

    monkeypatch.setattr(mem.settings, "memory_backend", "memory")
    monkeypatch.setattr(mem, "_store", None)
    yield


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Speech transport tests replace network calls and need no real key."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "test-placeholder")
