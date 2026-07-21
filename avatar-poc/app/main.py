from app.core.observability import configure_observability

configure_observability()

# Now safe to import the rest of the app -- observability is already wired.
from pathlib import Path

import logfire
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.avatar_routes import router as avatar_router
from app.api.transcription_routes import router as transcription_router
from app.api.voice_routes import router as voice_router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Avatar Voice POC", version="0.1.0")
    logfire.instrument_fastapi(app)
    app.include_router(voice_router)
    app.include_router(transcription_router)
    app.include_router(avatar_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "env": settings.app_env}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    logger.info("app initialized", extra={"node_name": "startup"})
    return app


app = create_app()
