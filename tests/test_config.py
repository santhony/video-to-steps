"""Settings field smoke tests for publish-related env vars."""

from __future__ import annotations

from pathlib import Path


def test_settings_have_publish_defaults(monkeypatch):
    # Make sure no .env in CWD leaks values into the test
    monkeypatch.delenv("PUBLISH_REPO", raising=False)
    monkeypatch.delenv("PUBLISH_BRANCH", raising=False)
    monkeypatch.delenv("PUBLISH_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLISH_CLONE_DIR", raising=False)
    monkeypatch.delenv("PUBLISH_ENABLED", raising=False)

    from config import Settings

    s = Settings(_env_file=None)
    assert s.publish_repo == "santhony/vts-publish"
    assert s.publish_branch == "main"
    assert s.publish_base_url == "https://santhony.github.io/vts-publish"
    assert s.publish_clone_dir == Path("data/publish_repo")
    assert s.publish_enabled is False


def test_settings_publish_repo_override(monkeypatch):
    monkeypatch.setenv("PUBLISH_REPO", "someone/elsewhere")
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    from config import Settings

    s = Settings(_env_file=None)
    assert s.publish_repo == "someone/elsewhere"
    assert s.publish_enabled is True


def test_settings_for_mode_local_is_passthrough(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8766")
    monkeypatch.setenv("CLOUD_LLM_BASE_URL", "https://api.deepseek.com")
    from config import Settings, settings_for_mode

    s = Settings(_env_file=None)
    out = settings_for_mode(s, "local")
    assert out.llm_base_url == "http://127.0.0.1:8766"


def test_settings_for_mode_cloud_overlays_set_fields(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "mlx_clip")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8766")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("VISION_BASE_URL", "http://127.0.0.1:8770")
    monkeypatch.setenv("CLOUD_EMBED_BACKEND", "jina_v4")
    monkeypatch.setenv("CLOUD_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CLOUD_LLM_API_KEY", "sk-cloud")
    monkeypatch.setenv("CLOUD_VISION_BASE_URL", "https://api.openai.com")
    from config import Settings, settings_for_mode

    s = Settings(_env_file=None)
    out = settings_for_mode(s, "cloud")
    assert out.embed_backend == "jina_v4"
    assert out.llm_base_url == "https://api.deepseek.com"
    assert out.llm_api_key == "sk-cloud"
    assert out.vision_base_url == "https://api.openai.com"
    # Unset cloud override falls back to base value
    assert out.llm_model == "deepseek-v4-flash"
