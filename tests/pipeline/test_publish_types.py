"""Smoke tests for the new publish types."""

from pathlib import Path

from pipeline.types import PublishError, StaticBundle


def test_static_bundle_is_a_dataclass():
    b = StaticBundle(html="<html></html>", file_map={"main.css": Path("static/css/main.css")})
    assert b.html == "<html></html>"
    assert b.file_map == {"main.css": Path("static/css/main.css")}


def test_static_bundle_has_slots():
    b = StaticBundle(html="", file_map={})
    # slots=True means assigning an unknown attribute raises
    try:
        b.unknown_attr = 1  # type: ignore[attr-defined]
    except AttributeError:
        return
    raise AssertionError("StaticBundle should have slots=True")


def test_publish_error_is_runtime_error():
    e = PublishError("git push failed")
    assert isinstance(e, RuntimeError)
    assert str(e) == "git push failed"


def test_manifest_publish_fields_default_to_none():
    """Existing manifests without these fields must continue to construct."""
    from pipeline.types import Manifest

    m = Manifest(job_id="abc", url="https://example.com")
    assert m.published_url is None
    assert m.published_at is None


def test_manifest_publish_fields_round_trip(tmp_path):
    """A manifest with a datetime survives write+read via the storage helpers."""
    from datetime import datetime, timezone
    from pipeline.storage import read_json, write_json_atomic
    from pipeline.types import Manifest

    m = Manifest(
        job_id="abc",
        url="https://example.com",
        published_url="https://santhony.github.io/vts-publish/abc/",
        published_at=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    p = tmp_path / "meta.json"
    write_json_atomic(p, m)
    loaded = read_json(p)
    assert loaded["published_url"] == "https://santhony.github.io/vts-publish/abc/"
    assert loaded["published_at"] == "2026-05-27T12:00:00+00:00"
