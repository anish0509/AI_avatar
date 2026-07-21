from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    llm_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"
    realtime_model: str = "gpt-realtime-2.1"
    realtime_voice: str = "marin"
    stt_model: str = "gpt-4o-transcribe"
    realtime_transcribe_model: str = "gpt-4o-transcribe"
    realtime_transcribe_language: str = "en"

    realtime_noise_reduction: str = "far_field"
    realtime_vad_threshold: float | None = None
    realtime_prefix_padding_ms: int | None = None
    realtime_silence_duration_ms: int | None = None
    realtime_transcribe_prompt: str = ""

    stt_debug_capture: bool = False

    llama_cloud_api_key: str = ""
    ingestion_input_dir: str = "data/input"
    ingestion_output_dir: str = "data/output"

    chunk_size: int = 1500
    chunk_overlap: int = 200

    pinecone_api_key: str = ""
    pinecone_index_name: str = "avatar-poc-content"
    pinecone_namespace: str = "avatar-poc"
    embedding_model: str = "llama-text-embed-v2"

    retrieval_top_k: int = 3
    retrieval_score_floor: float = 0.35

    answer_source: str = "gpt"

    memory_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = 86400

    logfire_token: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "avatar-poc"
    langsmith_tracing: str = "true"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    heygen_api_key: str = ""
    heygen_api_base: str = "https://api.liveavatar.com/v1"
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
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


settings = Settings()
