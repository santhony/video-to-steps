"""VisionCaptioner — caption a single frame via an OpenAI-shape chat endpoint.

Distinct from LLMClient because:
- Tighter max_tokens (captions are short).
- System prompt is read from prompts/vision_caption.md.
- Image is sent as a base64 data URL part inside the user message.

Per-frame failures (4xx/5xx from the provider, model refusal) bubble up as
exceptions; the caller (pipeline/caption_winners.py in Phase 5) catches
them and records caption=None for the affected frame.

pattern: Imperative Shell
This module orchestrates HTTP I/O (httpx SSE streams) and file I/O (prompt
reading, image base64 encoding). Pure regex logic is internal; the class
exposes only the async I/O interface.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class CaptionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "vision_caption.md"


def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _data_url(image: Path) -> str:
    raw = Path(image).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = image.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{b64}"


class VisionCaptioner:
    name: str

    def __init__(
        self,
        *,
        base_url: str,
        path: str,
        api_key: str,
        model: str,
        max_tokens: int = 300,
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

    async def caption(self, image: Path) -> CaptionResult:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Caption this frame."},
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    ],
                },
            ],
        }
        if self._include_usage:
            # Same caveat as LLMClient: some providers 400 on unknown
            # top-level params. Configurable via VISION_INCLUDE_USAGE.
            body["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        async with self._client.stream("POST", self._url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    # qwen-vl streaming-raw text path (unusual for vision but tolerate).
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

        text = _THINK_RE.sub("", "".join(text_parts)).strip()
        return CaptionResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def build_vision(settings: Any) -> "VisionCaptioner":
    return VisionCaptioner(
        base_url=settings.vision_base_url,
        path=settings.vision_path_chat,
        api_key=settings.vision_api_key,
        model=settings.vision_model,
        max_tokens=settings.vision_max_tokens,
        include_usage=settings.vision_include_usage,
    )
