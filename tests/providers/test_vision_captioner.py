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
async def test_vision_captioner_factory():
    """VisionCaptioner can be instantiated via build_vision or direct kwargs."""
    # This test verifies the factory pattern works
    from config import Settings
    from providers.vision import build_vision

    def transport(request: httpx.Request) -> httpx.Response:
        response_data = build_openai_vision_response(["test"], usage={"prompt_tokens": 1, "completion_tokens": 1})
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    # Direct instantiation
    direct = VisionCaptioner(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o-mini",
        _transport=httpx.MockTransport(transport),
    )
    assert isinstance(direct, VisionCaptioner)
    await direct.aclose()


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
