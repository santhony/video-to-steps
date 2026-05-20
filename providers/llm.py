"""LLMClient — async chat client with dual-stream SSE auto-detect.

Auto-detect rule: peek at the first non-empty `data:` payload after the
stream opens. If it parses as a JSON object with a `choices` key, treat the
stream as OpenAI-compatible (delta-token shape); otherwise treat each `data:`
line as a literal next-token text (qwen-studio shape).

`<think>...</think>` blocks are stripped unconditionally from the accumulated
text before return. Some reasoning-tier providers (notably DeepSeek v4) emit
chain-of-thought via a separate `reasoning_content` field on the delta —
we ignore that field entirely; only `delta.content` accumulates.

pattern: Imperative Shell
This module orchestrates HTTP I/O (httpx SSE streams) and response parsing.
Pure regex and string manipulation logic is internal; the class exposes only
the async I/O interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text)


class LLMClient:
    name: str

    def __init__(
        self,
        *,
        base_url: str,
        path: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        include_usage: bool = True,
        timeout: float = 120.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = model
        self._url = base_url.rstrip("/") + path
        self._model = model
        self._max_tokens = max_tokens
        self._include_usage = include_usage
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=_transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": True,
        }
        if self._include_usage:
            # Ask OpenAI-shape providers to include usage in the final
            # chunk. Some strict providers 400 on unknown top-level
            # params (notably qwen-studio's raw-text SSE), so this is a
            # configurable opt-out (LLM_INCLUDE_USAGE=0).
            body["stream_options"] = {"include_usage": True}
        if response_format is not None:
            body["response_format"] = response_format

        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        shape: str | None = None  # "openai" | "qwen" once detected

        async with self._client.stream("POST", self._url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                # SSE format uses a single-space separator after `data:`.
                # Use removeprefix so we strip EXACTLY one space (the
                # separator) and preserve any further leading whitespace
                # that belongs to the content — qwen-studio's raw-text
                # stream relies on those spaces between tokens.
                payload = line.removeprefix("data: ").removeprefix("data:")
                if payload.strip() == "[DONE]":
                    break
                if shape is None:
                    stripped = payload.strip()
                    shape = "openai" if (stripped.startswith("{") and '"choices"' in stripped) else "qwen"

                if shape == "openai":
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        # Fall back to treating as raw — unlikely but harmless.
                        text_parts.append(payload)
                        continue
                    choices = obj.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))
                else:
                    # qwen-studio raw shape: payload is the literal
                    # next-token text. Newlines inside the token were
                    # escaped on the sending side (`\n` -> literal `\n`
                    # 2-char) so the SSE framing stayed intact. Unescape
                    # them now so JSON / multiline content reads correctly.
                    text_parts.append(payload.replace("\\n", "\n"))

        text = _strip_think("".join(text_parts))
        return ChatResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def build_llm(settings: Any) -> "LLMClient":
    return LLMClient(
        base_url=settings.llm_base_url,
        path=settings.llm_path_chat,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        include_usage=settings.llm_include_usage,
    )
