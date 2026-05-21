"""Test-session-wide fixtures.

Configures Logfire in local-only mode (no token, nothing sent anywhere)
once, before any test runs. Without this, every logfire.span()/info() call
sprinkled through the service modules emits a LogfireNotConfiguredWarning
on every test that exercises them -- tests still pass either way (calling
Logfire before configure() doesn't raise, it just drops the data), but this
keeps output clean and doubles as a real check that the app's logfire calls
execute without error once configured.

Tests import service modules directly (e.g. `from app.api.voice_routes
import voice_ws`), never through app/main.py, so app/main.py's own
configure_observability() call never runs during pytest -- this fixture is
what stands in for it in the test environment.
"""

import logfire
import pytest


def pytest_configure(config) -> None:
    logfire.configure(token=None, send_to_logfire=False)


@pytest.fixture(autouse=True)
def _pin_in_memory_conversation_store(monkeypatch):
    """Pins conversation memory to the in-process store for every test, and
    resets the module-level `_store` singleton around each one.

    Pinning the backend explicitly (rather than inheriting whatever `.env`
    happens to say) is the same fix already applied to `answer_source` in the
    orchestration tests, for the same reason: a test suite whose behavior
    changes when a developer edits `.env` isn't testing anything reliably.
    Setting MEMORY_BACKEND=redis in .env really did turn 8 of these tests red.

    The singleton reset is what makes that pin effective. `_get_store()`
    caches one store for the process, so without this the FIRST test to touch
    memory would fix the backend for the whole session -- and a cached
    RedisConversationStore is worse than merely wrong here: `redis.asyncio`
    binds its connection to the event loop that created it, while
    pytest-asyncio gives every test a NEW loop. The cached client's transport
    dies with the loop that made it, so the next test to write hits a closed
    socket (`_ProactorSocketTransport closing ...`). That is purely a
    pytest-lifecycle artifact, NOT a production bug: uvicorn runs one loop per
    worker for the life of the process, so there the singleton is correct.

    Tests that specifically exercise backend selection (`_build_store` with
    memory_backend="redis") still work -- they call `_build_store()` directly,
    bypassing the singleton, and re-apply their own monkeypatch on top of this
    one.
    """
    from app.services import conversation_memory as mem

    monkeypatch.setattr(mem.settings, "memory_backend", "memory")
    monkeypatch.setattr(mem, "_store", None)
    yield
