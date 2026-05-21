from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    llm_model: str = "gpt-4o-mini"
    # Separate from llm_model (even though it defaults to the same value) --
    # Multimodal_RAG's own reference planner documented gpt-4o-mini
    # misrouting technical questions to CONVERSATIONAL and skipping
    # retrieval. Kept cheap by default here; bump independently if the same
    # misrouting shows up in practice, without touching the responder model.
    planner_model: str = "gpt-4o-mini"
    realtime_model: str = "gpt-realtime-2.1"
    realtime_voice: str = "marin"
    stt_model: str = "gpt-4o-transcribe"
    # gpt-4o-transcribe, NOT gpt-4o-mini-transcribe: on real mic audio the
    # mini model frequently mis-transcribed short, VAD-segmented words into
    # other scripts (e.g. "JavaScript" -> Urdu "جاوا اسکرپٹ"); the full model
    # transcribes the same clips correctly. See bug-report-streaming-stt.md.
    realtime_transcribe_model: str = "gpt-4o-transcribe"
    # Bias transcription toward a language. "en" for the current English
    # demo; set to "" (auto-detect) or "hi" when working with the project's
    # eventual Hindi/Hinglish content. Empty string -> omit, let the model
    # auto-detect.
    realtime_transcribe_language: str = "en"

    # --- Bug 4 (trailing phantom-token) mitigation knobs -----------------
    # All five default to empty/None so the transcription session behaves
    # EXACTLY as before (server_vad with API defaults) until a value is
    # explicitly set. Each is omitted from the session.update payload when
    # empty/None so OpenAI applies its own default. These are the levers the
    # Phase 2 experiments toggle one at a time -- see bug-report-streaming-stt.md.
    #
    # audio.input.noise_reduction.type: "" | "near_field" | "far_field".
    # Default "far_field": Phase 2 experiments (2026-07-13) on real captured mic
    # audio showed this cuts the Bug-4 trailing phantom token ~2/3 (trailing 33%
    # -> 11% macro-averaged) with NO accuracy or latency cost, fully eliminating
    # it on the clearest cases (api/javascript/react: 100% -> 0%). far_field
    # suits a laptop/room mic; use "near_field" for a headset held close, or ""
    # to disable. See bug-report-streaming-stt.md and scripts/stt_experiment.py.
    realtime_noise_reduction: str = "far_field"
    # turn_detection.threshold: VAD speech-probability cutoff (API default
    # 0.5). Higher (0.55-0.65) makes quiet non-speech less likely to be
    # captured into the segment tail. None -> omit.
    realtime_vad_threshold: float | None = None
    # turn_detection.prefix_padding_ms: audio kept BEFORE speech onset (API
    # default 300). Guardrail against clipping the first word when threshold
    # is raised -- not itself a hallucination lever. None -> omit.
    realtime_prefix_padding_ms: int | None = None
    # turn_detection.silence_duration_ms: silence needed to close a segment
    # (API default 500). NOTE: raising this was already shown to break live
    # streaming (Bug 3 wrong-turn, 700ms never finalized); lower values trim
    # the trailing near-silence this bug feeds on. None -> omit.
    realtime_silence_duration_ms: int | None = None
    # transcription.prompt: free-text decoding bias (weak lever). "" -> omit.
    realtime_transcribe_prompt: str = ""

    # Temporary diagnostic: when true, /ws/transcribe writes the exact PCM it
    # receives from the browser to tmp/stt_capture_<session>.wav so the raw
    # mic audio can be inspected. Off by default; not for production.
    stt_debug_capture: bool = False

    # --- Ingestion / document parsing (LlamaParse) ------------------------
    # First stage of the RAG ingestion pipeline (parse -> chunk -> embed):
    # PDFs dropped in ingestion_input_dir get parsed to Markdown + a metadata
    # sidecar in ingestion_output_dir. See app/ingestion/loaders/pdf_loader.py.
    llama_cloud_api_key: str = ""
    ingestion_input_dir: str = "data/input"
    ingestion_output_dir: str = "data/output"

    # --- Ingestion / chunking ----------------------------------------------
    # Stage 2: splits parsed Markdown into retrieval-sized pieces. Target
    # size matches Multimodal_RAG's proven default (1500 chars / ~300-400
    # tokens, 200 char overlap) -- see app/ingestion/chunking.py.
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # --- Ingestion / vector store (Pinecone) --------------------------------
    # Stage 3: embeds chunks server-side via Pinecone integrated inference
    # (no local embedding model, no torch) and upserts into a dedicated
    # index. Deliberately SEPARATE from ai-avatar-backend's ai-avatar-content
    # index -- that one is reserved for the production sales-coaching
    # knowledge base; this POC's general-document demo corpus stays clearly
    # apart from it. See app/ingestion/vector_store.py.
    pinecone_api_key: str = ""
    pinecone_index_name: str = "avatar-poc-content"
    pinecone_namespace: str = "avatar-poc"
    embedding_model: str = "llama-text-embed-v2"

    # --- Retrieval (query time) ---------------------------------------------
    # Dense search only for now (no reranking) -- see app/services/retrieval.py.
    # Reranking/hybrid search etc. are a later optimization pass, added once
    # this simple grounded-QA path is proven working end to end.
    retrieval_top_k: int = 3
    # Deterministic grounding gate: below this top-chunk similarity score,
    # stream_rag_response() abstains WITHOUT calling the LLM, rather than
    # trusting the system prompt alone to refuse -- confirmed in practice
    # that gpt-4o-mini answers well-known general questions (e.g. "what is
    # sales") from its own pretrained knowledge even when the retrieved
    # context is irrelevant. Calibrated on this corpus's real scores: clear
    # off-topic queries ("what is sales" 0.27, "what is mobile" 0.18) vs.
    # genuinely on-topic ones ("what is OLAP" 0.58, neural-network question
    # 0.54) -- 0.35 sits with margin in the gap between them. Revisit once
    # more real queries have been tried.
    retrieval_score_floor: float = 0.35

    # "gpt": plain stream_llm_response (no retrieval, current default).
    # "rag": stream_rag_response (retrieves top chunks from Pinecone, grounds
    # the answer in them). See app/services/rag_stream.py and
    # app/api/voice_routes.py, where this toggle is read.
    answer_source: str = "gpt"

    # --- Conversation memory (Phase 2, 2026-07-30) --------------------------
    # "memory": today's in-process dict (InMemoryConversationStore) -- lost on
    # restart, broken across --workers 2+. Default, and what every existing
    # test runs against.
    # "redis": RedisConversationStore -- survives restart, works across
    # workers. Deliberately NOT the default: it requires `docker-compose up`
    # (see avatar-poc/docker-compose.yml), which this project didn't need
    # before this feature. See app/services/conversation_memory.py.
    memory_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    # Sliding TTL, reset on every append_message() -- an actively-used
    # conversation never expires mid-use; only a thread with no new message
    # for this long is deleted. 86400s = 24h, per explicit request (not the
    # 1h first proposed).
    memory_ttl_seconds: int = 86400

    # --- Observability (Logfire + LangSmith) --------------------------------
    # Both fully optional -- the app behaves identically without tokens
    # configured, just without traces reaching either dashboard (verified
    # empirically: Logfire no-ops with token=""/None; LangSmith's @traceable
    # no-ops with no API key set). See app/core/observability.py for wiring
    # and why LangSmith needs an explicit os.environ forward despite this
    # project's "config only via Settings, never read os.environ directly"
    # rule -- the third-party langsmith SDK reads its own config via
    # os.environ internally, so Settings stays the single source of truth
    # that gets forwarded, rather than letting that SDK read .env itself.
    logfire_token: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "avatar-poc"
    langsmith_tracing: str = "true"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- HeyGen LiveAvatar (streaming lip-synced avatar) --------------------
    # Capability-test integration: our own TTS audio (currently
    # RealtimeApiSpeaker's PCM16/24kHz output) is pushed to HeyGen's LiveAvatar
    # LITE mode over a plain WebSocket, and HeyGen renders lip-synced video --
    # HeyGen never synthesizes speech itself, so there is no voice_id here.
    # See app/services/heygen_streaming.py.
    heygen_api_key: str = ""
    heygen_api_base: str = "https://api.liveavatar.com/v1"
    # Sandbox default: free, zero credits, but locked to this one avatar and
    # ~1 minute per session -- fine for a capability test. To go live, get a
    # real avatar_id (a UUID) from the LiveAvatar dashboard and set
    # HEYGEN_IS_SANDBOX=false.
    heygen_avatar_id: str = "dd73ea75-1218-4ef3-92ce-606d5f7fbc0a"
    heygen_is_sandbox: bool = True
    heygen_video_quality: str = "high"  # very_high | high | medium | low
    heygen_video_encoding: str = "H264"  # H264 | VP8
    heygen_session_timeout_s: float = 30.0

    app_env: str = "local"
    log_level: str = "INFO"

    @field_validator(
        "realtime_vad_threshold",
        "realtime_prefix_padding_ms",
        "realtime_silence_duration_ms",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        # A blank line in .env (e.g. REALTIME_VAD_THRESHOLD=) arrives as ""
        # -> treat it as "unset" so the field is omitted, rather than a
        # validation error on parsing "" to float/int.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


settings = Settings()
