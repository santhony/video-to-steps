# video-to-steps Implementation Plan — Phase 2: Provider implementations

**Goal:** Concrete `LLMClient`, `VisionCaptioner`, `JinaEmbedder`, and
`MlxClipEmbedder` classes that talk to real APIs, report token usage, and
behave correctly under the dual-stream SSE shapes used by qwen-studio and
OpenAI-compatible endpoints.

**Architecture:** Each provider class is a small async HTTP client built on
`httpx.AsyncClient`. The `LLMClient` auto-detects response shape by inspecting
the first non-empty `data:` line of the SSE stream (JSON vs raw text). The
`JinaEmbedder` posts to `/v1/embeddings` with a mixed text+image `input`
array. `MlxClipEmbedder` is import-guarded: factory call raises
`RuntimeError` on hosts without `mlx_clip` installed.

**Tech Stack:** Python 3.11+, `httpx` (async HTTP + SSE), `Pillow` (read frame
dimensions for vision messages), `numpy` (already added in Phase 1).

**Scope:** 2 of 7 phases.

**Codebase verified:** 2026-05-18. Phase 1 deliverables expected:
`config.Settings`, `pipeline/types.py`, `pipeline/storage.py`, `pricing.py`,
`providers/{llm,vision,embed}.py` (protocols + NotImplementedError stubs).

**External dependency findings (per research, see /tmp/plan-2026-05-18-vts-v1-29d408fa/findings.md):**
- ✓ Jina v4 endpoint: `POST https://api.jina.ai/v1/embeddings`, Bearer auth,
  mixed `input` array accepting `{"text": "..."}` and `{"image": "..."}`
  objects, vectors L2-normalized by default, `usage.total_tokens` reported.
- ✓ qwen-studio: bind 127.0.0.1:8766, `/chat`, raw `data: <text>` SSE,
  terminator `data: [DONE]`, NO `usage` reporting (we record zero billable
  tokens — local model, cost-only-zero is acceptable).
- ✓ OpenAI SSE: `data: {"choices":[{"delta":{"content":"..."}}], ...}`,
  terminator `data: [DONE]`. Usage in streaming requires
  `stream_options.include_usage=true`; arrives in final chunk with
  `choices: []` and a populated `usage` object.
- ✓ DeepSeek v4 reasoning models emit `reasoning_content` on the delta (not
  inline `<think>` tags in `content`). Defensive `<think>...</think>`
  stripping still applied because some providers (qwen) may inline.
- ✓ `response_format={"type":"json_object"}` supported by OpenAI / DeepSeek /
  Together. NOT by Anthropic (uses tools). Unknown for qwen-studio — we
  send it regardless; providers that ignore the field still work, and the
  outline pass has a slice-fallback in Phase 4.
- ✓ `mlx_clip`: github.com/harperreed/mlx_clip, Apple-Silicon-only, no PyPI
  wheel. L2 normalization not documented — `MlxClipEmbedder` normalizes
  client-side to be safe.
- ✓ Image base64: `data:image/jpeg;base64,<...>` — universally supported by
  OpenAI / Anthropic / qwen-vl in `image_url` part shape.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC4: Provider abstractions are config-only switchable
- **vts-v1.AC4.1 Success:** `LLMClient` chats successfully against both qwen-studio `/chat` (single-text SSE) and an OpenAI-shape `/v1/chat/completions` endpoint, distinguishing them by auto-detect on response shape.
- **vts-v1.AC4.2 Success:** `LLMClient.chat()` always strips `<think>...</think>` blocks from returned text.
- **vts-v1.AC4.3 Success:** `JinaEmbedder.embed_images()` and `.embed_texts()` return L2-normalized float32 vectors of consistent dimensionality.
- **vts-v1.AC4.4 Success:** Switching `EMBED_BACKEND` from `jina_v4` to `mlx_clip` requires only an env-var change; `mlx_clip` raises a clear actionable error on a host without the dep installed.
- **vts-v1.AC4.5 Success:** `VisionCaptioner.caption()` returns a 1–2 sentence caption focused on actions, tools, and materials for a checked-in test still frame.

---

<!-- START_TASK_1 -->
### Task 1: Bump requirements for HTTP + image handling

**Files:**
- Modify: `requirements.txt` (append)

**Implementation:**

Append two lines:

```
httpx>=0.27
Pillow>=10.0
```

`httpx` is the async HTTP client used by all three providers. `Pillow` is
needed by `VisionCaptioner` to read frame bytes and (later in Phase 3) by
`imagehash`.

**Verification:**

```bash
source venv/bin/activate
uv pip install -r requirements-dev.txt
python -c "import httpx, PIL; print(httpx.__version__, PIL.__version__)"
```
Expected: two version strings printed.

**Commit:**

```bash
git add requirements.txt
git commit -m "chore(vts-v1): add httpx + Pillow for providers"
```
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-3) -->

<!-- START_TASK_2 -->
### Task 2: `providers/llm.py` — concrete `LLMClient`

**Verifies:** vts-v1.AC4.1, vts-v1.AC4.2

**Files:**
- Modify: `providers/llm.py` (replace stub with concrete class; keep
  protocol shape and `ChatResult` dataclass from Phase 1)

**Implementation:**

The class:
1. Builds an `httpx.AsyncClient` per call (or one shared at construction
   time — choose construction-time for connection reuse).
2. POSTs to `{base_url}{path}` with `stream=True`.
3. For OpenAI shape, always sends `stream_options={"include_usage": true}`
   so the final chunk carries token counts.
4. Iterates the SSE stream line-by-line. For each `data: ...` line:
   - If the payload is the literal `[DONE]`, stop.
   - Otherwise, try `json.loads`. If parsing succeeds AND the object has a
     `choices` key, the stream is OpenAI-shape: extract
     `choices[0].delta.content` (may be `None` for the first/last chunks)
     and append. If the final chunk has `usage`, record it.
   - If parsing fails, treat the payload as raw token text (qwen-studio
     shape) and append it directly.
5. After the stream ends, strip `<think>...</think>` blocks (regex,
   non-greedy, dot-all) from the accumulated text and return `ChatResult`.
6. If `response_format` was requested AND we never got a usage chunk
   (qwen-studio path), token counts default to 0.

```python
"""LLMClient — async chat client with dual-stream SSE auto-detect.

Auto-detect rule: peek at the first non-empty `data:` payload after the
stream opens. If it parses as a JSON object with a `choices` key, treat the
stream as OpenAI-compatible (delta-token shape); otherwise treat each `data:`
line as a literal next-token text (qwen-studio shape).

`<think>...</think>` blocks are stripped unconditionally from the accumulated
text before return. Some reasoning-tier providers (notably DeepSeek v4) emit
chain-of-thought via a separate `reasoning_content` field on the delta —
we ignore that field entirely; only `delta.content` accumulates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


# `ChatResult` is the dataclass kept in this file from Phase 1; the
# concrete `LLMClient` below replaces the Phase-1 Protocol stub.

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
    ) -> None:
        self.name = model
        self._url = base_url.rstrip("/") + path
        self._model = model
        self._max_tokens = max_tokens
        self._include_usage = include_usage
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
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
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if shape is None:
                    shape = "openai" if (payload.startswith("{") and '"choices"' in payload) else "qwen"

                if shape == "openai":
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        # Fall back to treating as raw — unlikely but harmless.
                        text_parts.append(payload)
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))
                else:  # qwen-studio: payload is the literal next-token text
                    text_parts.append(payload)

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
```

**Note on file structure:** The protocol + `ChatResult` from Phase 1 stay in
the same file; the concrete class is appended in place of the
`NotImplementedError` factory. Restructure so the file ends up with:
1. The `ChatResult` dataclass (kept).
2. The concrete `LLMClient` class (replaces the protocol).
3. The `build_llm()` factory (replaces the stub).

Type-checker note: since the concrete `LLMClient` satisfies the `Protocol` it
replaces, deleting the `Protocol` class is fine — callers will type-check
against the concrete class.

**Testing:**

Tests must verify each AC case listed above:
- **vts-v1.AC4.1:** Build two fake SSE responses — one OpenAI shape, one
  qwen-studio shape — and assert `LLMClient.chat()` extracts the correct
  text from each. Use `httpx.MockTransport` to inject canned responses.
- **vts-v1.AC4.2:** Feed a fake OpenAI-shape response whose
  `delta.content` chunks contain `<think>internal</think>` markers; assert
  the returned `text` has zero `<think>` substrings.

Create `tests/__init__.py` (empty) and `tests/providers/__init__.py`
(empty) before writing the test files.

Test file: `tests/providers/test_llm_client.py` (unit).

Task-implementor generates concrete pytest code from the AC descriptions
and project test conventions (pytest + pytest-asyncio from Phase 1).

**Verification:**

```bash
source venv/bin/activate
pytest tests/providers/test_llm_client.py -v
```
Expected: all tests pass.

**Commit:**

```bash
git add providers/llm.py tests/
git commit -m "feat(vts-v1): LLMClient with dual-stream SSE auto-detect + <think>-strip"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `scripts/smoke_llm.py` — real-API smoke

**Verifies:** vts-v1.AC4.1 (against a real configured endpoint).

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/smoke_llm.py`

**Implementation:**

A simple async script that:
1. Loads `Settings`.
2. Builds an `LLMClient`.
3. Sends a one-message chat: `[{"role": "user", "content": "Say 'pong' and nothing else."}]`.
4. Prints `text`, `prompt_tokens`, `completion_tokens`, and a
   `pricing.compute_chat_cost(...)` estimate.

This script is operator-facing; failures should print a readable diagnostic
(missing API key, connection refused) without traceback noise. Wrap the
call in try/except and re-raise after a one-line summary.

```python
"""Smoke-test the configured LLM endpoint.

Usage: python scripts/smoke_llm.py

Reads .env via config.Settings; calls LLMClient once; prints token counts
and a price-table-derived cost estimate. Non-zero exit on failure.
"""

from __future__ import annotations

import asyncio
import sys

from config import get_settings
from pricing import compute_chat_cost
from providers.llm import build_llm


async def main() -> int:
    settings = get_settings()
    llm = build_llm(settings)
    try:
        result = await llm.chat([{"role": "user", "content": "Say 'pong' and nothing else."}])
    except Exception as exc:  # noqa: BLE001 — operator-facing one-liner is the point
        print(f"smoke_llm FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await llm.aclose()

    cost = compute_chat_cost(settings.llm_model, result.prompt_tokens, result.completion_tokens)
    print(f"text:              {result.text.strip()!r}")
    print(f"prompt_tokens:     {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

**Verification:**

If a Mode C `.env` is configured (DeepSeek or similar):

```bash
source venv/bin/activate
python scripts/smoke_llm.py
```
Expected: prints non-empty text, non-zero prompt/completion tokens, non-zero
cost estimate.

If no `.env` configured, the script exits 1 with a connection or auth error
— that's correct behavior; the smoke is only meaningful when a real endpoint
is provided.

**Commit:**

```bash
git add scripts/
git commit -m "feat(vts-v1): smoke_llm.py one-call diagnostic"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: `prompts/vision_caption.md` + `providers/vision.py`

**Verifies:** vts-v1.AC4.5

**Files:**
- Create: `prompts/__init__.py` (NOT needed; prompts are read as files)
- Create: `prompts/vision_caption.md`
- Modify: `providers/vision.py` (replace stub with concrete class)
- Create: `tests/providers/test_vision_captioner.py` (unit)
- Create: `tests/providers/fixtures/test_frame.jpg` (small checked-in test still)

**Implementation:**

`prompts/vision_caption.md` is plain markdown read at construction time and
treated as the system prompt. Keep it terse and grounded:

```markdown
You describe a single still frame from an instructional video.

Output one detailed sentence (or two if essential) that names the actions,
tools, and materials visibly present. Do NOT speculate about steps before or
after this frame. Do NOT name people. Do NOT describe lighting, mood, or
camera angle. Focus on what a viewer would need to know to identify which
step of a how-to this frame illustrates.
```

`providers/vision.py` — concrete class:

```python
"""VisionCaptioner — caption a single frame via an OpenAI-shape chat endpoint.

Distinct from LLMClient because:
- Tighter max_tokens (captions are short).
- System prompt is read from prompts/vision_caption.md.
- Image is sent as a base64 data URL part inside the user message.

Per-frame failures (4xx/5xx from the provider, model refusal) bubble up as
exceptions; the caller (pipeline/caption_winners.py in Phase 5) catches
them and records caption=None for the affected frame.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


# `CaptionResult` is the dataclass kept in this file from Phase 1; the
# concrete `VisionCaptioner` below replaces the Phase-1 Protocol stub.

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
    ) -> None:
        self.name = model
        self._url = base_url.rstrip("/") + path
        self._model = model
        self._max_tokens = max_tokens
        self._include_usage = include_usage
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
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
                choices = obj.get("choices") or []
                if choices:
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
```

**Note on file structure:** Same pattern as Task 2's `providers/llm.py`.
The rewritten `providers/vision.py` should end up with:
1. The `CaptionResult` dataclass (kept from Phase 1, redefined inline as
   shown above so no self-import is needed).
2. The concrete `VisionCaptioner` class (replaces the Phase-1 Protocol
   stub).
3. The `build_vision()` factory (replaces the Phase-1
   `NotImplementedError` stub).

For the test fixture, check in `tests/providers/fixtures/test_frame.jpg` —
a small (≤50 KB) image showing a hand holding a knife over a cutting board,
or similar instructional frame. Source it from a public-domain image bank
or generate one with ffmpeg from any short royalty-free video; the only
requirements are that it be checked in (so tests are reproducible) and
that it have visible action/tools/materials so the AC4.5 caption assertion
is meaningful.

**Testing:**

Tests must verify:
- **vts-v1.AC4.5:** Mock an OpenAI-shape SSE response that yields the
  string `"A hand grips a chef's knife on a wooden cutting board next to halved onions."` — assert `caption()` returns that text and ≥1 prompt token (from the mocked usage chunk). The real-endpoint check happens in the smoke script (Task 5).

Test file: `tests/providers/test_vision_captioner.py` (unit).

**Verification:**

```bash
source venv/bin/activate
pytest tests/providers/test_vision_captioner.py -v
```
Expected: tests pass.

**Commit:**

```bash
git add prompts/ providers/vision.py tests/providers/
git commit -m "feat(vts-v1): VisionCaptioner + caption system prompt + fixture frame"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `scripts/smoke_vision.py` — real-API smoke

**Files:**
- Create: `scripts/smoke_vision.py`

**Implementation:**

Mirrors `smoke_llm.py`. Captions the checked-in `tests/providers/fixtures/test_frame.jpg`:

```python
"""Smoke-test the configured vision endpoint with a checked-in test frame."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config import get_settings
from pricing import compute_vision_cost
from providers.vision import build_vision


FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "providers" / "fixtures" / "test_frame.jpg"


async def main() -> int:
    settings = get_settings()
    vis = build_vision(settings)
    try:
        result = await vis.caption(FIXTURE)
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_vision FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await vis.aclose()

    cost = compute_vision_cost(settings.vision_model, result.prompt_tokens, result.completion_tokens)
    print(f"caption:           {result.text.strip()!r}")
    print(f"prompt_tokens:     {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

**Verification:**

With a Mode C `.env` configured against gpt-4o-mini or similar:

```bash
source venv/bin/activate
python scripts/smoke_vision.py
```
Expected: prints a 1–2 sentence caption mentioning actions/tools/materials,
non-zero prompt/completion tokens, non-zero cost.

**Commit:**

```bash
git add scripts/smoke_vision.py
git commit -m "feat(vts-v1): smoke_vision.py one-call diagnostic"
```
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 6-8) -->

<!-- START_TASK_6 -->
### Task 6: `providers/embed_jina.py` — `JinaEmbedder`

**Verifies:** vts-v1.AC4.3

**Files:**
- Create: `providers/embed_jina.py`
- Create: `tests/providers/test_jina_embedder.py` (unit)

**Implementation:**

Posts to `https://api.jina.ai/v1/embeddings` with a mixed `input` array.
For images, content is sent as base64 data URLs (mirroring vision-caption
shape) so the same encoding works regardless of input type. Batches
according to `settings.jina_embed_batch`. Defensively L2-normalizes every
returned vector (Jina normalizes by default, but normalizing again is a
no-op and guards against config drift).

```python
"""JinaEmbedder — multimodal embeddings via Jina /v1/embeddings.

Both image and text inputs go through the same endpoint; the request body's
`input` array carries `{"text": "..."}` or `{"image": "<data-url-or-url>"}`
objects. Vectors come back float32 (cast on receipt) and L2-normalized
(re-normalized client-side defensively).

Batches are sized by `settings.jina_embed_batch` to stay within the API's
per-request limits; results are concatenated in input order.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from providers.embed import EmbedResult


def _data_url(image: Path) -> str:
    raw = Path(image).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = image.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{b64}"


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero for any zero vectors (shouldn't happen, but defensive).
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


class JinaEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jina-embeddings-v4",
        batch: int = 64,
        base_url: str = "https://api.jina.ai",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._batch = batch
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_batch(self, inputs: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
        body = {
            "model": self._model,
            "input": inputs,
            "normalized": True,
        }
        resp = await self._client.post(self._url, json=body)
        resp.raise_for_status()
        data = resp.json()
        rows = [item["embedding"] for item in data.get("data", [])]
        vectors = np.asarray(rows, dtype=np.float32)
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens", 0))
        return vectors, tokens

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        if not paths:
            return EmbedResult(vectors=np.zeros((0, 0), dtype=np.float32), billable_tokens=0)

        all_vectors: list[np.ndarray] = []
        total_tokens = 0
        for i in range(0, len(paths), self._batch):
            chunk = paths[i : i + self._batch]
            inputs = [{"image": _data_url(p)} for p in chunk]
            vecs, tokens = await self._post_batch(inputs)
            all_vectors.append(vecs)
            total_tokens += tokens

        vectors = np.concatenate(all_vectors, axis=0)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=total_tokens)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=np.zeros((0, 0), dtype=np.float32), billable_tokens=0)

        all_vectors: list[np.ndarray] = []
        total_tokens = 0
        for i in range(0, len(texts), self._batch):
            chunk = texts[i : i + self._batch]
            inputs = [{"text": t} for t in chunk]
            vecs, tokens = await self._post_batch(inputs)
            all_vectors.append(vecs)
            total_tokens += tokens

        vectors = np.concatenate(all_vectors, axis=0)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=total_tokens)
```

**Testing:**

Tests must verify:
- **vts-v1.AC4.3 — images:** Mock the Jina endpoint to return a 3-vector
  response (synthetic 2048-d float lists with non-trivial magnitudes).
  Assert returned `vectors.dtype == np.float32`, `vectors.shape ==
  (3, 2048)`, and `np.allclose(np.linalg.norm(vectors, axis=1), 1.0)`.
- **vts-v1.AC4.3 — texts:** Same as above for `embed_texts`.
- **vts-v1.AC4.3 — consistent dim:** Call both `embed_images` and
  `embed_texts` against the same mocked endpoint; assert
  `images.vectors.shape[1] == texts.vectors.shape[1]`.

Test file: `tests/providers/test_jina_embedder.py` (unit).

**Verification:**

```bash
source venv/bin/activate
pytest tests/providers/test_jina_embedder.py -v
```
Expected: all tests pass.

**Commit:**

```bash
git add providers/embed_jina.py tests/providers/test_jina_embedder.py
git commit -m "feat(vts-v1): JinaEmbedder with batching + defensive L2-normalize"
```
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: `providers/embed_mlx_clip.py` — import-guarded `MlxClipEmbedder`

**Verifies:** vts-v1.AC4.4

**Files:**
- Create: `providers/embed_mlx_clip.py`
- Create: `tests/providers/test_mlx_clip_factory.py` (unit)

**Implementation:**

The class itself attempts to import `mlx_clip` at module-construction time
(inside `__init__`), so importing this MODULE never fails — only
instantiation. The factory in Task 8 invokes this only when
`EMBED_BACKEND=mlx_clip`. The class implements `embed_images` /
`embed_texts` against the (Apple-Silicon-only) library; on the v1
acceptance host (Linux), the factory raises `RuntimeError` and we never get
this far.

Per research: `mlx_clip` does not document L2 normalization; normalize
client-side.

```python
"""MlxClipEmbedder — local multimodal embeddings via mlx_clip (Apple Silicon).

This module is import-safe everywhere (you can `import providers.embed_mlx_clip`
on Linux without error). Instantiation requires `mlx_clip` to be available;
the factory in providers/embed.py raises a clear RuntimeError on hosts
without the dep.

This class is NOT exercised in v1's acceptance smoke test. It exists so the
README's "try Mode A on Macbook" path works without code changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from providers.embed import EmbedResult


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


class MlxClipEmbedder:
    def __init__(self, *, model: str = "openai/clip-vit-base-patch32") -> None:
        try:
            import mlx_clip  # noqa: F401 — proves availability
        except ImportError as exc:
            raise RuntimeError(
                "mlx_clip not installed. Mode A requires Apple Silicon + "
                "`pip install git+https://github.com/harperreed/mlx_clip`. "
                "On non-Apple-Silicon hosts use EMBED_BACKEND=jina_v4."
            ) from exc

        self._model_id = model
        # Hold the actual mlx_clip handle; exact API surface verified at
        # integration time. Treat the import success as a green light.
        self._mlx_clip = __import__("mlx_clip")

    def name(self) -> str:
        return f"mlx_clip:{self._model_id}"

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        # mlx_clip's API is synchronous; we run it inline since v1 does not
        # exercise this path under load. If/when Mode A becomes a test
        # target, wrap in asyncio.to_thread.
        rows = [self._mlx_clip.image_encoder(str(p)) for p in paths]
        vectors = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        rows = [self._mlx_clip.text_encoder(t) for t in texts]
        vectors = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)
```

**Testing:**

Tests must verify (on Linux / no-`mlx_clip` host):
- **vts-v1.AC4.4:** Calling `MlxClipEmbedder()` raises `RuntimeError`
  whose message references `mlx_clip not installed` and
  `EMBED_BACKEND=jina_v4`.

Test file: `tests/providers/test_mlx_clip_factory.py` (unit).

The build-embedder factory test (also AC4.4) is in Task 8.

**Verification:**

```bash
source venv/bin/activate
pytest tests/providers/test_mlx_clip_factory.py -v
```
Expected: tests pass.

**Commit:**

```bash
git add providers/embed_mlx_clip.py tests/providers/test_mlx_clip_factory.py
git commit -m "feat(vts-v1): MlxClipEmbedder import-guarded for Mode A (unexercised in v1)"
```
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Embedder factory + factory tests + smoke script

**Verifies:** vts-v1.AC4.3, vts-v1.AC4.4 (factory routing).

**Files:**
- Modify: `providers/embed.py` (replace `build_embedder` stub with real
  factory; preserve `Embedder` and `FrameExtractor` protocols and
  `EmbedResult`).
- Create: `scripts/smoke_embed.py`
- Create: `tests/providers/test_embed_factory.py` (unit)

**Implementation:**

`build_embedder` reads `settings.embed_backend` and dispatches:

```python
"""Embedder + FrameExtractor protocols and the embedder factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class EmbedResult:
    vectors: np.ndarray   # shape (n, d), dtype float32, L2-normalized
    billable_tokens: int


class Embedder(Protocol):
    def name(self) -> str: ...
    async def embed_images(self, paths: list[Path]) -> EmbedResult: ...
    async def embed_texts(self, texts: list[str]) -> EmbedResult: ...


class FrameExtractor(Protocol):
    def name(self) -> str: ...
    def extract(self, video: Path, out_dir: Path) -> list:
        """Returns list[pipeline.types.Frame]. Implementations land in Phase 3."""
        ...


def build_embedder(settings: Any) -> Embedder:
    backend = (settings.embed_backend or "").strip().lower()
    if backend in ("jina_v4", "jina", "jina-v4"):
        from providers.embed_jina import JinaEmbedder
        return JinaEmbedder(
            api_key=settings.jina_api_key,
            model=settings.jina_model,
            batch=settings.jina_embed_batch,
        )
    if backend in ("mlx_clip", "mlx-clip", "mlx"):
        from providers.embed_mlx_clip import MlxClipEmbedder  # may raise at instantiation
        return MlxClipEmbedder(model=settings.mlx_clip_model)
    raise ValueError(
        f"Unknown EMBED_BACKEND={settings.embed_backend!r}. "
        "Valid values: jina_v4, mlx_clip."
    )
```

`scripts/smoke_embed.py`:

```python
"""Smoke-test the configured embedder.

Embeds one text and one frame (the checked-in fixture) and prints the
output dimensionality, vector dtype, L2-norm of the first row, and a
cost estimate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

from config import get_settings
from pricing import compute_embed_cost
from providers.embed import build_embedder


FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "providers" / "fixtures" / "test_frame.jpg"


async def main() -> int:
    settings = get_settings()
    emb = build_embedder(settings)
    try:
        text_res = await emb.embed_texts(["a hand chopping an onion"])
        image_res = await emb.embed_images([FIXTURE])
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_embed FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        aclose = getattr(emb, "aclose", None)
        if aclose is not None:
            await aclose()

    total_tokens = text_res.billable_tokens + image_res.billable_tokens
    cost = compute_embed_cost(emb.name(), total_tokens)
    print(f"backend:           {emb.name()}")
    print(f"text dim:          {text_res.vectors.shape}")
    print(f"image dim:         {image_res.vectors.shape}")
    print(f"text L2:           {float(np.linalg.norm(text_res.vectors[0])):.6f}")
    print(f"image L2:          {float(np.linalg.norm(image_res.vectors[0])):.6f}")
    print(f"billable_tokens:   {total_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

**Testing:**

Tests must verify:
- **vts-v1.AC4.4 — Jina path:** Stub `settings.embed_backend = "jina_v4"`
  and `settings.jina_api_key = "test"`; assert `build_embedder(settings)`
  returns a `JinaEmbedder` (use `isinstance`). Do not actually call the
  endpoint.
- **vts-v1.AC4.4 — mlx_clip path on Linux:** Stub
  `settings.embed_backend = "mlx_clip"`; assert `build_embedder(settings)`
  raises `RuntimeError` referencing `mlx_clip not installed`.
- **vts-v1.AC4.4 — unknown backend:** Stub
  `settings.embed_backend = "totally-bogus"`; assert `build_embedder`
  raises `ValueError` whose message lists valid values.

Test file: `tests/providers/test_embed_factory.py` (unit).

**Verification:**

```bash
source venv/bin/activate
pytest tests/providers/ -v
```
Expected: ALL Phase 2 unit tests pass.

If a Mode C `.env` is configured:

```bash
python scripts/smoke_embed.py
```
Expected: backend `jina-embeddings-v4`, text and image shapes both
`(1, N)` with matching N (default 2048), both L2 norms ≈ 1.0, non-zero
billable tokens, non-zero cost.

**Commit:**

```bash
git add providers/embed.py scripts/smoke_embed.py tests/providers/test_embed_factory.py
git commit -m "feat(vts-v1): embedder factory routing + smoke_embed.py"
```
<!-- END_TASK_8 -->

<!-- END_SUBCOMPONENT_C -->
