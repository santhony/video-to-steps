"""Unit tests for VisionCaptioner — caption a frame via OpenAI-shape endpoint."""

from __future__ import annotations

import json
import pytest
import httpx
from pathlib import Path

from providers.vision import VisionCaptioner, CaptionResult


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test_frame.jpg"


def build_openai_vision_response(content_chunks: list[str], usage: dict | None = None) -> bytes:
    """Build an OpenAI-shape SSE response for vision caption."""
    lines: list[str] = []
    for chunk in content_chunks:
        payload = json.dumps({"choices": [{"delta": {"content": chunk}}]})
        lines.append(f"data: {payload}\n")
    if usage is not None:
        usage_json = json.dumps({"choices": [], "usage": usage})
        lines.append(f"data: {usage_json}\n")
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode("utf-8")


@pytest.mark.asyncio
async def test_vision_captioner_returns_caption():
    """AC4.5: VisionCaptioner.caption() returns caption text from mocked response."""
    expected_caption = "A hand grips a chef's knife on a wooden cutting board next to halved onions."
    response_data = build_openai_vision_response([expected_caption], usage={"prompt_tokens": 42, "completion_tokens": 15})

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    captioner = VisionCaptioner(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o-mini",
        _transport=httpx.MockTransport(transport),
    )

    result = await captioner.caption(FIXTURE)

    assert result.text == expected_caption
    assert result.prompt_tokens == 42
    assert isinstance(result, CaptionResult)
    await captioner.aclose()


@pytest.mark.asyncio
async def test_vision_captioner_usage_counting():
    """VisionCaptioner extracts usage counts from mocked response."""
    response_data = build_openai_vision_response(
        ["A caption."],
        usage={"prompt_tokens": 42, "completion_tokens": 15}
    )

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    captioner = VisionCaptioner(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o-mini",
        _transport=httpx.MockTransport(transport),
    )

    result = await captioner.caption(FIXTURE)

    assert result.prompt_tokens == 42
    assert result.completion_tokens == 15
    await captioner.aclose()


@pytest.mark.asyncio
async def test_vision_captioner_strips_think_tags():
    """VisionCaptioner strips <think>...</think> blocks from caption."""
    response_data = build_openai_vision_response(
        ["A hand ", "<think>internal reasoning here</think>", "grips a knife."],
        usage={"prompt_tokens": 20, "completion_tokens": 10}
    )

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    captioner = VisionCaptioner(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o-mini",
        _transport=httpx.MockTransport(transport),
    )

    result = await captioner.caption(FIXTURE)

    assert result.text == "A hand grips a knife."
    assert "<think>" not in result.text
    await captioner.aclose()


@pytest.mark.asyncio
async def test_vision_captioner_factory_via_build_vision():
    """build_vision factory constructs VisionCaptioner from Settings."""
    from config import Settings
    from providers.vision import build_vision

    # Build a Settings instance with test configuration.
    # _env_file=None isolates the test from any local .env so it doesn't
    # inadvertently pick up the operator's real provider values.
    settings = Settings(
        _env_file=None,
        vision_api_key="test-key",
        vision_model="gpt-4o-mini",
        vision_base_url="https://api.openai.com",
        vision_path_chat="/v1/chat/completions",
    )

    captioner = build_vision(settings)
    assert isinstance(captioner, VisionCaptioner)
    assert captioner.name == "gpt-4o-mini"
    await captioner.aclose()


@pytest.mark.asyncio
async def test_vision_captioner_reads_system_prompt():
    """VisionCaptioner reads system prompt from prompts/vision_caption.md."""
    from providers.vision import _system_prompt

    prompt = _system_prompt()
    assert "instructional video" in prompt
    assert len(prompt) > 0


@pytest.mark.asyncio
async def test_vision_captioner_handles_non_list_choices():
    """I-2: If choices field is non-list (e.g., string), no AttributeError is raised."""
    # Build a response where choices is a string instead of a list
    payload = json.dumps({"choices": "stringy"})
    response_data = f"data: {payload}\ndata: [DONE]\n".encode("utf-8")

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    captioner = VisionCaptioner(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o-mini",
        _transport=httpx.MockTransport(transport),
    )

    result = await captioner.caption(FIXTURE)

    # Should return empty text without raising
    assert result.text == ""
    await captioner.aclose()
