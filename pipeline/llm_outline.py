"""LLM Pass 1 — coarse outline of the transcript into StepOutlines.

Robust against providers that ignore `response_format=json_object`: after
the chat call, we try strict `json.loads(text)` first, then fall through to
a slice-fallback that locates the outermost `[...]` array and parses that.

pattern: Imperative Shell
This module orchestrates I/O (LLM chat calls) with functional core logic
(prompt formatting, JSON parsing, transcript formatting). The core functions
are pure and testable without I/O; the async orchestration handles the chat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from providers.llm import LLMClient
from .types import Cue, StepOutline, TokenUsage
from ._prompts import load_system_user


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "outline.md"


def _format_transcript(cues: list[Cue]) -> str:
    return "\n".join(f"[{c.start:.2f}–{c.end:.2f}] {c.text}" for c in cues)


def _slice_first_array(text: str) -> str | None:
    """Locate the first '[' and its matching ']' (depth-aware) and return that substring."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_outline(text: str) -> list[dict[str, Any]]:
    """Returns a list[dict] with index/start/end/brief; raises ValueError if unrecoverable.

    Two-stage parsing:
    1. Strict json.loads first — handles both the prompt-shape
       `{"steps":[...]}` (when the provider honored response_format) AND
       a bare `[...]` (when it returned just the array).
    2. Slice fallback — when prose surrounds the JSON (qwen-studio, or any
       provider that drifted), locate the first balanced `[...]` and parse
       that. If the model wrapped `{"steps":[...]}` in prose, the inner
       `[...]` is still what we want, so this path is correct for both
       prompt-shape and bare-array responses.
    """
    text = text.strip()
    # Stage 1: strict JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            return obj["steps"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Stage 2: slice fallback.
    sliced = _slice_first_array(text)
    if sliced is not None:
        try:
            arr = json.loads(sliced)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    raise ValueError(f"could not parse outline JSON from response: {text[:200]!r}...")


async def llm_outline(cues: list[Cue], llm: LLMClient) -> tuple[list[StepOutline], TokenUsage]:
    """Calls the LLM once to divide cues into StepOutlines."""
    sys_prompt, user_template = load_system_user(_PROMPT_PATH)
    user_prompt = user_template.format(transcript=_format_transcript(cues))

    result = await llm.chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        # Outline is structured JSON extraction, not a reasoning task.
        # qwen-studio + ds4-server honor `think=false` to skip the
        # chain-of-thought trace — saves the bulk of generation time
        # on local reasoning models. Cloud providers ignore the field.
        extra_body={"think": False},
    )

    raw_steps = _parse_outline(result.text)
    outlines: list[StepOutline] = []
    for i, s in enumerate(raw_steps):
        # Defense against None: if brief is None/missing, default to ""; if non-string, str() it.
        brief_val = s.get("brief")
        if brief_val is None or isinstance(brief_val, str):
            brief = (brief_val or "").strip()
        else:
            brief = str(brief_val).strip()
        outlines.append(
            StepOutline(
                index=int(s.get("index", i)),
                start=float(s["start"]),
                end=float(s["end"]),
                brief=brief,
            )
        )
    # Re-sort by index, then start, to defend against models that drift.
    outlines.sort(key=lambda o: (o.index, o.start))
    return outlines, TokenUsage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
