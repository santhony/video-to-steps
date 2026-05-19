"""LLMClient protocol + factory stub.

Concrete implementation lands in Phase 2. The contract:
- `chat()` takes a list of OpenAI-style messages and returns the text plus
  prompt/completion token counts so the orchestrator can sum cost.
- `response_format` is an optional hint for providers that support
  `{"type": "json_object"}`; clients tolerate providers that ignore it.
- Returned text is ALWAYS stripped of any `<think>...</think>` reasoning
  blocks the model may emit inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    name: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...


def build_llm(settings: Any) -> LLMClient:
    """Returns a configured LLMClient. Implemented in Phase 2."""
    raise NotImplementedError("LLMClient factory implemented in Phase 2")
