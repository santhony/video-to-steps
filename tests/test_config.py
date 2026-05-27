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
