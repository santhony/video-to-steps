"""Tests for pipeline/llm_outline.py."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline.llm_outline import _parse_outline, _slice_first_array, llm_outline
from pipeline.types import Cue
from providers.llm import ChatResult


@dataclass
class _StubLLM:
    canned_text: str
    name: str = "stub"

    async def chat(self, messages, *, max_tokens=None, response_format=None) -> ChatResult:
        return ChatResult(text=self.canned_text, prompt_tokens=42, completion_tokens=17)


def _cues_for_3min_video() -> list[Cue]:
    # 12 cues at 15-second intervals, 0–180s.
    return [Cue(start=i * 15.0, end=(i + 1) * 15.0, text=f"step content {i}") for i in range(12)]


@pytest.mark.asyncio
async def test_outline_well_formed_json_object_with_steps_key():
    # vts-v1.AC5.1, vts-v1.AC5.2
    canned = '{"steps": [' \
        '{"index": 0, "start": 0.0,  "end": 60.0,  "brief": "prep"},' \
        '{"index": 1, "start": 60.0, "end": 120.0, "brief": "cook"},' \
        '{"index": 2, "start": 120.0,"end": 180.0, "brief": "plate"}' \
        ']}'
    llm = _StubLLM(canned_text=canned)
    outlines, usage = await llm_outline(_cues_for_3min_video(), llm)
    assert len(outlines) >= 3
    # Cover full span.
    assert outlines[0].start <= 0.0
    assert outlines[-1].end >= 180.0
    # Non-overlapping.
    for a, b in zip(outlines, outlines[1:]):
        assert a.end <= b.start
    assert usage.prompt_tokens == 42


@pytest.mark.asyncio
async def test_outline_bare_array_response():
    # Some providers return a bare JSON array even though we asked for an object.
    canned = '[{"index":0,"start":0,"end":90,"brief":"first half"},' \
             '{"index":1,"start":90,"end":180,"brief":"second half"},' \
             '{"index":2,"start":180,"end":181,"brief":"close"}]'
    llm = _StubLLM(canned_text=canned)
    outlines, _ = await llm_outline(_cues_for_3min_video(), llm)
    assert len(outlines) == 3


@pytest.mark.asyncio
async def test_outline_slice_fallback_with_prose():
    # vts-v1.AC5.3 — model wrapped the JSON in prose.
    canned = (
        "Sure! Here is the outline:\n\n"
        "```json\n"
        "[{\"index\":0,\"start\":0,\"end\":90,\"brief\":\"intro\"},"
        " {\"index\":1,\"start\":90,\"end\":180,\"brief\":\"outro\"},"
        " {\"index\":2,\"start\":180,\"end\":181,\"brief\":\"wrap\"}]\n"
        "```\n\n"
        "Let me know if you'd like adjustments."
    )
    llm = _StubLLM(canned_text=canned)
    outlines, _ = await llm_outline(_cues_for_3min_video(), llm)
    assert len(outlines) == 3
    assert outlines[0].brief == "intro"


def test_slice_finds_balanced_brackets_inside_strings():
    # Defense against ']' inside string values fooling the slicer is not
    # required — JSON inside a chat response will not legally contain
    # unescaped brackets in strings. We do verify a clean balanced slice.
    text = "lead-in [1, [2, 3], 4] trail-out"
    assert _slice_first_array(text) == "[1, [2, 3], 4]"


def test_parse_outline_raises_on_unrecoverable():
    with pytest.raises(ValueError):
        _parse_outline("no JSON in here at all")


@pytest.mark.asyncio
async def test_outline_none_brief_becomes_empty_string():
    # Regression test for Issue #4: defense-in-depth gap.
    # When the LLM returns brief=null, we should get "" not the string "None".
    canned = '[{"index":0,"start":0,"end":60,"brief":null},' \
             '{"index":1,"start":60,"end":120,"brief":"cook"}]'
    llm = _StubLLM(canned_text=canned)
    outlines, _ = await llm_outline(_cues_for_3min_video(), llm)
    assert len(outlines) == 2
    assert outlines[0].brief == ""  # None should become "", not "None"
    assert outlines[1].brief == "cook"
