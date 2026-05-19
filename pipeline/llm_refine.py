"""LLM Pass 2 — per-step polished imperative text.

Concurrency is bounded by `asyncio.Semaphore(max_in_flight)` to respect
provider rate limits. Per-step failures degrade gracefully: the step
falls back to its brief text rather than aborting the job.

pattern: Imperative Shell
This module orchestrates I/O (LLM chat calls) with functional core logic
(prompt formatting, JSON parsing). The core functions are pure and testable
without I/O; the async orchestration handles concurrency and error recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from providers.llm import LLMClient
from .types import Cue, Frame, Step, StepOutline, TokenUsage
from ._prompts import load_system_user


log = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "refine.md"


def _cues_in_window(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """Functional Core: Select cues that overlap or are near the [start, end] window."""
    pad = 1.0
    return [c for c in cues if c.end >= (start - pad) and c.start <= (end + pad)]


def _format_cues(cues: list[Cue]) -> str:
    """Functional Core: Format a list of cues into prompt text."""
    if not cues:
        return "(no cues in window)"
    return "\n".join(f"[{c.start:.2f}–{c.end:.2f}] {c.text}" for c in cues)


def _format_captions(captions: list[str | None]) -> str:
    """Functional Core: Format frame captions into prompt text."""
    nonempty = [c for c in captions if c]
    if not nonempty:
        return "(no captions available for this step)"
    return "\n".join(f"- {c}" for c in nonempty)


def _parse_instruction(text: str) -> str:
    """Functional Core: Extract the `instruction` field from a JSON-object response.

    Falls back through stages:
    1. Strict json.loads on entire text
    2. Slice fallback — find balanced { ... } and try to parse
    3. Return raw text (should not occur if model honored response_format)
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("instruction"), str):
            return obj["instruction"].strip()
    except json.JSONDecodeError:
        pass

    # Slice fallback — find balanced { ... } and try.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and isinstance(obj.get("instruction"), str):
                            return obj["instruction"].strip()
                    except json.JSONDecodeError:
                        break
                    break

    # Last resort — return the cleaned text.
    return text.strip()


async def _refine_one(
    outline: StepOutline,
    cues: list[Cue],
    winners: list[Frame],
    captions: list[str | None],
    llm: LLMClient,
    sem: asyncio.Semaphore,
    sys_prompt: str,
    user_template: str,
) -> tuple[Step, TokenUsage]:
    """Imperative Shell: Refine a single step via LLM call with per-step error recovery."""
    async with sem:
        user_prompt = user_template.format(
            brief=outline.brief,
            cues=_format_cues(_cues_in_window(cues, outline.start, outline.end)),
            captions=_format_captions(captions),
        )
        try:
            result = await llm.chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # 1500 covers the visible content (1-3 sentences, ~150 tokens)
                # PLUS the hidden reasoning_content that DeepSeek-v4 and similar
                # reasoning models emit (often 500-1200 tokens). Trimming this
                # too low yields empty content. Cheap models can override via
                # settings.llm_max_tokens.
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            instruction = _parse_instruction(result.text) or outline.brief
            usage = TokenUsage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_refine step %d failed: %s; falling back to brief.", outline.index, exc)
            instruction = outline.brief
            usage = TokenUsage()

        return (
            Step(
                index=outline.index,
                start=outline.start,
                end=outline.end,
                instruction=instruction,
                frames=list(winners),
            ),
            usage,
        )


async def llm_refine(
    outlines: list[StepOutline],
    cues: list[Cue],
    winners_by_step: dict[int, list[Frame]],
    captions: dict[int, str | None],
    llm: LLMClient,
    *,
    max_in_flight: int = 4,
) -> tuple[list[Step], TokenUsage]:
    """Imperative Shell: Refine each StepOutline into a Step with polished instruction text.

    Concurrent fanout is bounded by max_in_flight semaphore to respect provider rate limits.
    Per-step failures degrade gracefully (fallback to brief), rather than aborting the run.
    """
    sys_prompt, user_template = load_system_user(_PROMPT_PATH)
    sem = asyncio.Semaphore(max_in_flight)

    tasks: list[asyncio.Task[tuple[Step, TokenUsage]]] = []
    for o in outlines:
        ws = winners_by_step.get(o.index, [])
        caps = [captions.get(f.index) for f in ws]
        tasks.append(
            asyncio.create_task(
                _refine_one(o, cues, ws, caps, llm, sem, sys_prompt, user_template)
            )
        )

    results = await asyncio.gather(*tasks)
    steps = [r[0] for r in results]
    steps.sort(key=lambda s: s.index)
    total = TokenUsage()
    for _, u in results:
        total.prompt_tokens += u.prompt_tokens
        total.completion_tokens += u.completion_tokens
    return steps, total
