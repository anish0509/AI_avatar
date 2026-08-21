# Avatar Voice

A FastAPI demo for streamed voice answers and a lip-synced video avatar. Type or speak a question in the voice page, or use the separate LiveAvatar page for video responses.

The answer pipeline supports three modes:

- `gpt`: direct OpenAI responses.
- `rag`: answers grounded in documents indexed in Pinecone.
- `agent`: retrieval planning and conversation history for follow-up questions.

## Run locally

Use Python 3.12. From the repository root:

```sh
cd avatar-poc
python3.12 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the server:

```sh
uvicorn app.main:app --reload
```

Open [the voice page](http://localhost:8000/) or [the avatar page](http://localhost:8000/static/avatar.html). Restart the server after changing `.env`. Microphone access requires localhost or HTTPS.

The server and `/health` can start without API keys; provider-backed requests require the relevant credentials. The video page also needs `HEYGEN_API_KEY` and an avatar ID available to your LiveAvatar account. Sandbox sessions have limited duration, so check your account's limits before testing long answers.

## Containers

From `avatar-poc`, copy `.env.example` to `.env` and configure the required keys.

```sh
# Redis for a server running on the host
docker compose up -d

# API, Redis, and Caddy on http://localhost
docker compose -f docker-compose.prod.yml up -d --build

# HTTPS on a host with DNS pointing at it and ports 80/443 reachable
SITE_ADDRESS=avatar.example.com docker compose -f docker-compose.prod.yml up -d --build
```

The image defaults to one API worker. The full stack uses two workers with Redis for shared conversation history. Redis is internal to the full stack; the development Redis port binds only to localhost. Redis data and Caddy certificates use named volumes.

```sh
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml down
```

This demo has no authentication or rate limiting. Add access controls before exposing paid API routes to untrusted users. Existing avatar connections close during a redeploy.

## Documents and memory

Set `LLAMA_CLOUD_API_KEY` for PDF parsing and `PINECONE_API_KEY` for indexing and retrieval. Place PDFs in `avatar-poc/data/input`, then run from `avatar-poc`:

```sh
python -m scripts.parse_documents
python -m scripts.chunk_documents
python -m scripts.ingest_documents
```

Select `ANSWER_SOURCE=rag` or `ANSWER_SOURCE=agent` to use the indexed content. Below `RETRIEVAL_SCORE_FLOOR`, the retrieval pipeline declines to answer from the retrieved documents. Tune that threshold for your corpus.

Only `agent` mode uses conversation history. Its default in-process store is lost on restart and requires a single worker. Set `MEMORY_BACKEND=redis` for persistent, shared history; `MEMORY_TTL_SECONDS` controls Redis expiry. A browser page reload starts a new conversation.

## Checks

```sh
cd avatar-poc
python -m pytest -q
```

Unit tests use local fakes and do not call paid services. The scripts named `check_*` perform manual integration checks and may use provider credits. Optional Logfire and LangSmith credentials enable remote tracing.

The PDF parser currently uses the deprecated `llama-cloud-services` SDK. Its migration is separate from the runtime fixes here; parsing should be verified with your account before ingesting a new corpus.

See [HISTORY.md](HISTORY.md) for the reconstructed commit dates.

## CI and local deployment

GitHub Actions runs the Python tests, checks JavaScript syntax, then builds and smoke-tests the API/Redis/Caddy stack on pushes to `main` and pull requests. You can also run it from the Actions tab. No API secrets are needed for CI.

For local deployment, install Docker with Compose 2.24.4 or newer and run from the repository root:

```sh
make deploy   # build, start, and wait for health checks
make status
make logs
make stop     # stop containers; preserve Redis data
```

Open http://localhost:8087. The deployment creates `avatar-poc/.env` from the example if missing; add your API keys there and run `make deploy` again. Existing `.env` files are preserved. The local proxy binds only to localhost.

Local delivery is manual: after pulling an update, run `make deploy`. GitHub Actions does not deploy to your laptop or require a self-hosted runner.
