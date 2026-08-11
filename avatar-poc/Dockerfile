FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.lock .
RUN pip install -r requirements.lock

RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
RUN mkdir -p tmp data/input data/output && chown -R appuser:appuser tmp data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Multiple workers require shared conversation storage; Compose configures Redis.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
