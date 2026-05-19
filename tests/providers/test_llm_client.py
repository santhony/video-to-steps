"""Unit tests for LLMClient — dual-stream SSE auto-detect and <think>-strip."""

from __future__ import annotations

import pytest
import httpx

from providers.llm import LLMClient, ChatResult


def build_openai_response(content_chunks: list[str], usage: dict | None = None) -> bytes:
    """Build an OpenAI-shape SSE response."""
    lines: list[str] = []
    for chunk in content_chunks:
        lines.append(f'data: {{"choices":[{{"delta":{{"content":"{chunk}"}}}}]}}\n')
    if usage is not None:
        import json
        usage_json = json.dumps(usage)
        lines.append(f'data: {{"choices":[], "usage": {usage_json}}}\n')
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode("utf-8")


def build_qwen_response(text_chunks: list[str]) -> bytes:
    """Build a qwen-studio raw-text SSE response."""
    lines: list[str] = [f"data: {chunk}\n" for chunk in text_chunks]
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode("utf-8")


@pytest.mark.asyncio
async def test_llm_client_openai_shape():
    """AC4.1: LLMClient detects and parses OpenAI-shape SSE."""
    response_data = build_openai_response(["hello", ""], usage=None)

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4",
        _transport=httpx.MockTransport(transport),
    )

    result = await client.chat([{"role": "user", "content": "test"}])

    assert result.text == "hello"
    assert isinstance(result, ChatResult)
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_qwen_shape():
    """AC4.1: LLMClient detects and parses qwen-studio raw-text SSE."""
    response_data = build_qwen_response(["hello", "world"])

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="http://127.0.0.1:8766",
        path="/chat",
        api_key="",
        model="qwen",
        _transport=httpx.MockTransport(transport),
    )

    result = await client.chat([{"role": "user", "content": "test"}])

    assert result.text == "helloworld"
    assert isinstance(result, ChatResult)
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_strips_think_tags():
    """AC4.2: LLMClient strips <think>...</think> blocks from response."""
    response_data = build_openai_response(
        ["Result is ", "<think>scratch</think>", "final"]
    )

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4",
        _transport=httpx.MockTransport(transport),
    )

    result = await client.chat([{"role": "user", "content": "test"}])

    assert result.text == "Result is final"
    assert "<think>" not in result.text
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_usage_counting():
    """AC4.2: LLMClient extracts and returns usage counts."""
    usage = {"prompt_tokens": 12, "completion_tokens": 34}
    response_data = build_openai_response(["hello"], usage=usage)

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="https://api.openai.com",
        path="/v1/chat/completions",
        api_key="test-key",
        model="gpt-4",
        _transport=httpx.MockTransport(transport),
    )

    result = await client.chat([{"role": "user", "content": "test"}])

    assert result.prompt_tokens == 12
    assert result.completion_tokens == 34
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_qwen_defaults_zero_tokens():
    """When qwen-studio provides no usage, counts default to 0."""
    response_data = build_qwen_response(["hello"])

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=response_data,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="http://127.0.0.1:8766",
        path="/chat",
        api_key="",
        model="qwen",
        _transport=httpx.MockTransport(transport),
    )

    result = await client.chat([{"role": "user", "content": "test"}])

    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    await client.aclose()
