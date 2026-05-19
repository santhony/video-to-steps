"""Runtime configuration.

Single source of truth for env-driven settings. Imported everywhere; never
mutated at runtime.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8090, alias="APP_PORT")
    jobs_root: Path = Field(default=Path("./data/jobs"), alias="JOBS_ROOT")

    # Embedder
    embed_backend: str = Field(default="jina_v4", alias="EMBED_BACKEND")
    jina_api_key: str = Field(default="", alias="JINA_API_KEY")
    jina_model: str = Field(default="jina-embeddings-v4", alias="JINA_MODEL")
    jina_embed_batch: int = Field(default=64, alias="JINA_EMBED_BATCH")
    mlx_clip_model: str = Field(default="openai/clip-vit-base-patch32", alias="MLX_CLIP_MODEL")

    # Text LLM
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_path_chat: str = Field(default="/v1/chat/completions", alias="LLM_PATH_CHAT")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    # Some OpenAI-strict providers (gpt-*, DeepSeek, Together) accept
    # stream_options={"include_usage": true} for token counts in the final
    # SSE chunk; others (qwen-studio, some local proxies) may 400 on
    # unknown top-level params. Default True; flip to False for strict
    # providers that reject it.
    llm_include_usage: bool = Field(default=True, alias="LLM_INCLUDE_USAGE")

    # Vision LLM
    vision_base_url: str = Field(default="https://api.openai.com", alias="VISION_BASE_URL")
    vision_path_chat: str = Field(default="/v1/chat/completions", alias="VISION_PATH_CHAT")
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")
    vision_model: str = Field(default="gpt-4o-mini", alias="VISION_MODEL")
    vision_max_tokens: int = Field(default=300, alias="VISION_MAX_TOKENS")
    vision_include_usage: bool = Field(default=True, alias="VISION_INCLUDE_USAGE")

    # Concurrency
    refine_max_in_flight: int = Field(default=4, alias="REFINE_MAX_IN_FLIGHT")
    caption_max_in_flight: int = Field(default=16, alias="CAPTION_MAX_IN_FLIGHT")

    # Feature flags
    whisper_fallback: bool = Field(default=False, alias="WHISPER_FALLBACK")


def get_settings() -> Settings:
    """Returns a fresh Settings instance. Callers may cache as needed."""
    return Settings()
