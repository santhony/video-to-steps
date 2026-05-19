"""Tests for pipeline/llm_refine.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline.llm_refine import llm_refine
from pipeline.types import Cue, Frame, StepOutline
from providers.llm import ChatResult


@dataclass
class _StubLLM:
    canned_per_index: dict[int, str]
    name: str = "stub"
    _call_idx: int = 0

    async def chat(self, messages, *, max_tokens=None, response_format=None) -> ChatResult:
        # Detect step index from the prompt body — system prompt instructs JSON.
        # For the test we just round-robin in arrival order.
        idx = self._call_idx
        self._call_idx += 1
        text = self.canned_per_index.get(idx, '{"instruction": "Default fallback step."}')
        return ChatResult(text=text, prompt_tokens=11, completion_tokens=7)


@pytest.mark.asyncio
async def test_refine_produces_imperative_sentences():
    # vts-v1.AC5.4 — assert each instruction is a populated string with
    # at least one sentence-ending period. Imperative-mood is a model
    # behavior; we assert structural correctness only.
    outlines = [
        StepOutline(index=0, start=0.0,  end=30.0,  brief="prep ingredients"),
        StepOutline(index=1, start=30.0, end=60.0,  brief="cook"),
        StepOutline(index=2, start=60.0, end=90.0,  brief="plate and serve"),
    ]
    cues = [Cue(start=i * 5.0, end=(i + 1) * 5.0, text=f"voice line {i}") for i in range(18)]
    winners = {
        0: [Frame(index=10, timestamp=10.0, path=Path("/tmp/0010.jpg"))],
        1: [Frame(index=40, timestamp=40.0, path=Path("/tmp/0040.jpg"))],
        2: [Frame(index=70, timestamp=70.0, path=Path("/tmp/0070.jpg"))],
    }
    captions = {10: "hands chopping carrots", 40: "boiling pot", 70: "plated dish"}
    canned = {
        0: '{"instruction": "Chop the carrots into thin rounds. Set them aside in a bowl."}',
        1: '{"instruction": "Bring the pot of water to a rolling boil over high heat."}',
        2: '{"instruction": "Plate the dish and serve immediately."}',
    }
    llm = _StubLLM(canned_per_index=canned)
    steps, usage = await llm_refine(
        outlines=outlines,
        cues=cues,
        winners_by_step=winners,
        captions=captions,
        llm=llm,
        max_in_flight=4,
    )
    assert len(steps) == 3
    for s in steps:
        assert s.instruction
        # 1-3 sentences: count period-terminators.
        sentences = [t for t in s.instruction.split(".") if t.strip()]
        assert 1 <= len(sentences) <= 3
        # Frames preserved on the Step.
        assert len(s.frames) == 1
    assert usage.prompt_tokens == 33  # 3 calls × 11


@pytest.mark.asyncio
async def test_refine_falls_back_when_chat_fails():
    class BoomLLM:
        name = "boom"
        async def chat(self, messages, *, max_tokens=None, response_format=None):
            raise RuntimeError("simulated 503")

    outlines = [StepOutline(index=0, start=0.0, end=10.0, brief="THE FALLBACK")]
    steps, usage = await llm_refine(
        outlines=outlines,
        cues=[],
        winners_by_step={0: []},
        captions={},
        llm=BoomLLM(),
        max_in_flight=1,
    )
    assert len(steps) == 1
    assert steps[0].instruction == "THE FALLBACK"
    assert usage.prompt_tokens == 0


@pytest.mark.asyncio
async def test_refine_parses_object_with_prose_around():
    canned = {
        0: 'Here is the JSON you asked for:\n```json\n{"instruction":"Heat the pan."}\n```\nDone.',
    }
    outlines = [StepOutline(index=0, start=0.0, end=10.0, brief="heat")]
    llm = _StubLLM(canned_per_index=canned)
    steps, _ = await llm_refine(
        outlines=outlines,
        cues=[],
        winners_by_step={0: []},
        captions={},
        llm=llm,
        max_in_flight=1,
    )
    assert steps[0].instruction == "Heat the pan."
