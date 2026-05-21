"""Wires up Logfire + LangSmith tracing. configure_observability() MUST be
called as the very first thing in main.py, before any other app module is
imported. Confirmed (Multimodal_RAG's own documented gotcha, and empirically
reproduced here) that Logfire silently no-ops for the rest of the process if
any module calls logfire.info()/span() before logfire.configure() has run --
it doesn't error, it just emits a LogfireNotConfiguredWarning and drops the
data. There's no way to "catch up" after the fact, so import order matters.

Both providers are fully optional: the app behaves identically without
tokens configured, just without traces reaching either dashboard. IMPORTANT,
confirmed the hard way (an actual crash caught before it shipped): calling
logfire.configure() with no token and send_to_logfire left at its default
(None -> "if-token-present") does NOT gracefully no-op -- it raises
LogfireConfigError demanding `logfire auth` or a token, crashing app startup
entirely. That's different from *never calling* configure() at all (which
genuinely does no-op, just with a warning). So send_to_logfire is set
EXPLICITLY here based on whether a token is configured, rather than trusting
the "if-token-present" default to degrade gracefully on its own.

LangSmith's own SDK reads its config (LANGSMITH_API_KEY etc.) via
os.environ internally -- a constraint of the third-party langsmith package,
not something we control. To keep Settings as the single source of truth
(per this project's "config only via BaseSettings, never read os.environ
directly" rule) rather than letting the langsmith SDK read .env on its own,
this module reads from `settings` and explicitly forwards into os.environ --
a write, not a read, so it satisfies that third-party constraint without
bypassing Settings as the one validated source of config.
"""

import os

import logfire

from app.core.config import settings


def configure_observability() -> None:
    logfire.configure(
        token=settings.logfire_token or None,
        send_to_logfire=bool(settings.logfire_token),
        service_name="avatar-poc",
    )

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
