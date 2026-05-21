# Python 3.12-slim matches the local venv (3.12.8), so a dependency that
# resolves differently on 3.11 can't surprise us only in production.
# slim (not alpine): the wheels this project needs are manylinux builds, and
# alpine's musl libc would force slow source builds of several of them.
FROM python:3.12-slim

# PYTHONUNBUFFERED is load-bearing, not cosmetic: this app logs structured
# JSON to stdout, and without it those lines sit in a buffer instead of
# reaching `docker logs` -- which makes a hung turn look like silence.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies are copied and installed BEFORE the app code so this layer is
# cached. App code changes constantly; requirements.txt rarely does. Copying
# everything at once would reinstall ~145 packages on every code edit.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# app/ carries app/static/ with it, which app/main.py resolves relative to
# its own location -- so the demo pages work without a separate COPY.
# scripts/ is included deliberately: check_heygen_session.py is the fastest
# way to diagnose an outbound-connectivity problem from INSIDE the container
# (`docker compose exec api python -m scripts.check_heygen_session`), which
# is exactly when you need it and can't reproduce it from your laptop.
COPY app ./app
COPY scripts ./scripts

# Non-root. Safe because nothing on the serving path writes to disk -- the
# only filesystem write in the running app is the STT debug capture
# (transcription_routes.py), which is behind STT_DEBUG_CAPTURE=false. The
# ingestion scripts DO write, but they don't run on this server; ingestion
# stays local and pushes to Pinecone's cloud.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# --workers 2 is only correct because MEMORY_BACKEND=redis is forced in
# docker-compose.prod.yml. With the in-process dict, two workers would each
# hold their own separate conversation history and follow-up questions would
# resolve against whichever worker happened to answer. Do not raise the
# worker count without confirming Redis is still the configured backend.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
