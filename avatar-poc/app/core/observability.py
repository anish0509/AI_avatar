"""Configure optional Logfire and LangSmith tracing before serving requests."""

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
