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
