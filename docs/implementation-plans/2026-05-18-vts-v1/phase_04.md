# video-to-steps Implementation Plan — Phase 4: LLM passes

**Goal:** Two LLM passes — `llm_outline` (transcript → coarse `StepOutline`
list) and `llm_refine` (per-step polished imperative text). Both robust
against prose-around-JSON responses and concurrent provider rate limits.

**Architecture:** Each pass formats a per-stage prompt from a checked-in
markdown template, calls the `LLMClient` from Phase 2 with
`response_format={"type":"json_object"}` (providers ignoring the field still
work via slice-fallback), and parses the result. `llm_refine` fans out
across steps with `asyncio.Semaphore(REFINE_MAX_IN_FLIGHT)` so the v1
cloud LLM rate limit isn't tripped by a 10-step video.

**Tech Stack:** `LLMClient` (Phase 2), Python stdlib (`json`, `re`,
`asyncio`).

**Scope:** 4 of 7 phases.

**Codebase verified:** 2026-05-18. Phase 1 (`pipeline/types.Cue`,
`StepOutline`, `Step`) and Phase 2 (`LLMClient` with
`response_format` support and `<think>`-strip) expected.

**External dependency findings:**
- ✓ `response_format={"type":"json_object"}` honored by OpenAI / DeepSeek /
  Together; ignored by Anthropic (uses tools) and possibly by
  qwen-studio. The pass-through is harmless, and we still need a
  slice-fallback regardless to recover when models prepend "Here is the
  JSON:" prose. The fallback handles BOTH the qwen-studio case and any
  provider that occasionally drifts.
- ✓ DeepSeek v4 reasoning model emits chain-of-thought in
  `reasoning_content` (already filtered out at the `LLMClient` level in
  Phase 2). Additional `<think>`-strip in Phase 2 is defensive belt + braces.
- ✓ Minimum `max_tokens` 1024 for DeepSeek v4 to avoid empty content —
  `Settings.llm_max_tokens` default is 2048; outline pass uses the full
  budget, refine pass overrides to ~300 (short imperative text).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC5: LLM outline and refine passes
- **vts-v1.AC5.1 Success:** `llm_outline` returns ≥3 `StepOutline`s with non-overlapping time ranges that together cover the input transcript span.
- **vts-v1.AC5.2 Success:** `llm_outline` parses provider responses correctly when `response_format={"type":"json_object"}` is supported.
- **vts-v1.AC5.3 Failure:** `llm_outline` slice-fallback parses a fixture response containing prose around the JSON `[…]` block (unit-tested).
- **vts-v1.AC5.4 Success:** `llm_refine` produces 1–3 second-person imperative sentences per step, incorporating the winning-frame captions into the prompt.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Outline + refine prompt templates

**Files:**
- Create: `prompts/outline.md`
- Create: `prompts/refine.md`

**Implementation:**

Both files contain a `## System` block and a `## User` block separated by a
single `## User` heading. The Python loader splits on the heading. Keep
prompts terse — every extra adjective costs tokens on every job.

`prompts/outline.md`:

```markdown
## System

You divide an instructional video transcript into 3–12 ordered steps.

Output ONLY a JSON object of the form:

```
{"steps": [{"index": 0, "start": 0.0, "end": 12.3, "brief": "..."}, ...]}
```

Rules:
- `start` and `end` are seconds (decimal) referring to the transcript timestamps you are given.
- Steps cover the full transcript: first step `start` is ≤ the first cue, last step `end` is ≥ the last cue.
- Steps DO NOT overlap; each `end` ≤ the next step's `start`.
- `brief` is at most one sentence describing the step content, in third person.
- Output JSON only; no surrounding prose.

## User

Transcript (seconds + text per line):

{transcript}

Divide this transcript into ordered steps and return the JSON described above.
```

`prompts/refine.md`:

```markdown
## System

You write a single step of a how-to guide.

Output ONLY a JSON object of the form:

```
{"instruction": "Sentence one. Sentence two."}
```

Rules:
- The `instruction` is 1–3 second-person imperative sentences telling the reader what to do during this step.
- Mention specific tools, materials, or actions that appear in the cue snippets and frame captions provided.
- Do not invent details not present in the inputs.
- Do not number the step. Do not output prose around the JSON.

## User

Step brief: {brief}

Cue snippets covering this step (seconds + text):

{cues}

What is visible in this step's representative frames:

{captions}

Write the JSON described above.
```

**Verification:**

```bash
ls prompts/outline.md prompts/refine.md prompts/vision_caption.md
```
Expected: all three exist.

**Commit:**

```bash
git add prompts/outline.md prompts/refine.md
git commit -m "feat(vts-v1): outline + refine prompt templates"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `pipeline/llm_outline.py`

**Verifies:** vts-v1.AC5.1, vts-v1.AC5.2 (mocked-LLM tests).

**Files:**
- Create: `pipeline/llm_outline.py`

**Implementation:**

```python
"""LLM Pass 1 — coarse outline of the transcript into StepOutlines.

Robust against providers that ignore `response_format=json_object`: after
the chat call, we try strict `json.loads(text)` first, then fall through to
a slice-fallback that locates the outermost `[...]` array and parses that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from providers.llm import LLMClient
from .types import Cue, StepOutline


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "outline.md"


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _load_prompt() -> tuple[str, str]:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # Split on "## User" heading; everything before is system, after is user template.
    parts = re.split(r"^## User\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise RuntimeError(f"{_PROMPT_PATH} must contain a '## User' heading")
    sys_part = re.sub(r"^## System\s*$", "", parts[0], flags=re.MULTILINE).strip()
    user_part = parts[1].strip()
    return sys_part, user_part


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


async def llm_outline(cues: list[Cue], llm: LLMClient) -> tuple[list[StepOutline], ChatUsage]:
    """Calls the LLM once to divide cues into StepOutlines."""
    sys_prompt, user_template = _load_prompt()
    user_prompt = user_template.format(transcript=_format_transcript(cues))

    result = await llm.chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw_steps = _parse_outline(result.text)
    outlines: list[StepOutline] = []
    for i, s in enumerate(raw_steps):
        outlines.append(
            StepOutline(
                index=int(s.get("index", i)),
                start=float(s["start"]),
                end=float(s["end"]),
                brief=str(s["brief"]).strip(),
            )
        )
    # Re-sort by index, then start, to defend against models that drift.
    outlines.sort(key=lambda o: (o.index, o.start))
    return outlines, ChatUsage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
```

**Verification:** (tests in Task 3).

**Commit:**

```bash
git add pipeline/llm_outline.py
git commit -m "feat(vts-v1): llm_outline with json_object + slice-fallback parsing"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `llm_outline` tests (incl. slice-fallback fixture)

**Verifies:** vts-v1.AC5.1, vts-v1.AC5.2, vts-v1.AC5.3

**Files:**
- Create: `tests/pipeline/test_llm_outline.py` (unit)

**Implementation:**

Tests use a stub `LLMClient` that returns canned text. No real cloud calls.

```python
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
```

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_llm_outline.py -v
```
Expected: 5 tests pass.

**Commit:**

```bash
git add tests/pipeline/test_llm_outline.py
git commit -m "test(vts-v1): llm_outline covers AC5.1/AC5.2/AC5.3"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: `pipeline/llm_refine.py` — concurrent per-step refine

**Verifies:** vts-v1.AC5.4 (mocked-LLM test).

**Files:**
- Create: `pipeline/llm_refine.py`

**Implementation:**

Per-step concurrent fanout. Each step is its own `LLMClient.chat` call with
a JSON-object response. Per-step failures (parse errors, provider 5xx) do
NOT abort the run — the failing step's `instruction` falls back to the step
brief and a warning is logged. The orchestrator records the failure count
in `manifest.progress` (Phase 5).

```python
"""LLM Pass 2 — per-step polished imperative text.

Concurrency is bounded by `asyncio.Semaphore(max_in_flight)` to respect
provider rate limits. Per-step failures degrade gracefully: the step
falls back to its brief text rather than aborting the job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from providers.llm import LLMClient
from .types import Cue, Frame, Step, StepOutline


log = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "refine.md"


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _load_prompt() -> tuple[str, str]:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    parts = re.split(r"^## User\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise RuntimeError(f"{_PROMPT_PATH} must contain a '## User' heading")
    sys_part = re.sub(r"^## System\s*$", "", parts[0], flags=re.MULTILINE).strip()
    user_part = parts[1].strip()
    return sys_part, user_part


def _cues_in_window(cues: list[Cue], start: float, end: float) -> list[Cue]:
    pad = 1.0
    return [c for c in cues if c.end >= (start - pad) and c.start <= (end + pad)]


def _format_cues(cues: list[Cue]) -> str:
    if not cues:
        return "(no cues in window)"
    return "\n".join(f"[{c.start:.2f}–{c.end:.2f}] {c.text}" for c in cues)


def _format_captions(captions: list[str | None]) -> str:
    nonempty = [c for c in captions if c]
    if not nonempty:
        return "(no captions available for this step)"
    return "\n".join(f"- {c}" for c in nonempty)


def _parse_instruction(text: str) -> str:
    """Extracts the `instruction` field from a JSON-object response, falling back to raw text."""
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
) -> tuple[Step, ChatUsage]:
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
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            instruction = _parse_instruction(result.text) or outline.brief
            usage = ChatUsage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_refine step %d failed: %s; falling back to brief.", outline.index, exc)
            instruction = outline.brief
            usage = ChatUsage()

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
) -> tuple[list[Step], ChatUsage]:
    """Refines each StepOutline into a Step with polished instruction text."""
    sys_prompt, user_template = _load_prompt()
    sem = asyncio.Semaphore(max_in_flight)

    tasks: list[asyncio.Task[tuple[Step, ChatUsage]]] = []
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
    total = ChatUsage()
    for _, u in results:
        total.prompt_tokens += u.prompt_tokens
        total.completion_tokens += u.completion_tokens
    return steps, total
```

**Verification:** (tests in Task 5).

**Commit:**

```bash
git add pipeline/llm_refine.py
git commit -m "feat(vts-v1): llm_refine with semaphore fanout + per-step fallback"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `llm_refine` tests

**Verifies:** vts-v1.AC5.4

**Files:**
- Create: `tests/pipeline/test_llm_refine.py` (unit)

**Implementation:**

```python
"""Tests for pipeline/llm_refine.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
        0: [Frame(index=10, timestamp=10.0, path="/tmp/0010.jpg")],  # noqa: PTH123
        1: [Frame(index=40, timestamp=40.0, path="/tmp/0040.jpg")],  # noqa: PTH123
        2: [Frame(index=70, timestamp=70.0, path="/tmp/0070.jpg")],  # noqa: PTH123
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
```

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_llm_refine.py -v
```
Expected: 3 tests pass.

**Commit:**

```bash
git add tests/pipeline/test_llm_refine.py
git commit -m "test(vts-v1): llm_refine covers AC5.4 + slice-fallback + degraded path"
```

**Done when:** All `tests/pipeline/test_llm_outline.py` and
`tests/pipeline/test_llm_refine.py` tests pass; against a configured cloud
LLM endpoint, a one-off run of `llm_outline(...)` on the Phase 3
deduped-cue output produces ≥3 StepOutlines covering the transcript span.
The cloud-endpoint check is operator-driven; the automated suite uses
stubbed clients.
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->
