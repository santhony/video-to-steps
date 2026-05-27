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
    # Optional explicit cache dir for mlx_clip's converted MLX weights.
    # Empty (default) → ~/.cache/video-to-steps/mlx_clip/<model-slug>.
    # First run downloads + converts from HF; subsequent runs reuse.
    mlx_clip_cache_dir: str = Field(default="", alias="MLX_CLIP_CACHE_DIR")
    # vision_caption (single-key) embedder: text-embedding endpoint that
    # consumes captions produced by the vision LLM. Empty base_url/api_key
    # → reuse the vision provider's base_url/api_key (OpenAI single-key
    # case). Path defaults to /v1/embeddings if blank.
    text_embed_base_url: str = Field(default="", alias="TEXT_EMBED_BASE_URL")
    text_embed_path: str = Field(default="", alias="TEXT_EMBED_PATH")
    text_embed_api_key: str = Field(default="", alias="TEXT_EMBED_API_KEY")
    text_embed_model: str = Field(default="", alias="TEXT_EMBED_MODEL")

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
    whisper_model: str = Field(default="base.en", alias="WHISPER_MODEL")

    # Publish to GitHub Pages
    publish_repo: str = Field(default="santhony/vts-publish", alias="PUBLISH_REPO")
    publish_branch: str = Field(default="main", alias="PUBLISH_BRANCH")
    publish_base_url: str = Field(
        default="https://santhony.github.io/vts-publish",
        alias="PUBLISH_BASE_URL",
    )
    publish_clone_dir: Path = Field(
        default=Path("data/publish_repo"), alias="PUBLISH_CLONE_DIR"
    )
    publish_enabled: bool = Field(default=False, alias="PUBLISH_ENABLED")


def get_settings() -> Settings:
    """Returns a fresh Settings instance. Callers may cache as needed."""
    return Settings()
