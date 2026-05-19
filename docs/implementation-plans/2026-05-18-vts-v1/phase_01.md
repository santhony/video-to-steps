# video-to-steps Implementation Plan — Phase 1: Foundation

**Goal:** Stand up the repo scaffold, configuration model, type system, and
provider protocols so subsequent phases have a stable skeleton to hang work
from. No external behavior yet.

**Architecture:** Single-process Python 3.11 service. Pydantic `Settings` is
the single source of truth for runtime config. Plain dataclasses model
internal pipeline types. Provider concerns live behind protocols
(`LLMClient`, `VisionCaptioner`, `Embedder`, `FrameExtractor`) so cloud↔local
swaps are config-only. Atomic JSON writes (write-then-rename) prevent torn
reads by the status-poll endpoint.

**Tech Stack:** Python 3.11+, uv (venv + dep install), Pydantic v2 / pydantic-settings, python-dotenv (via pydantic-settings), standard library only for the rest of Phase 1.

**Scope:** 1 of 7 phases (Foundation).

**Codebase verified:** 2026-05-18.

**Codebase verification findings:**
- ✓ Greenfield worktree: only `README.md` (stub), `PLAN.md`, `.gitignore`,
  `docs/design-plans/2026-05-18-vts-v1.md`. No app code.
- ✓ `.gitignore` already contains `venv/`, `data/`, `__pycache__/`, `*.pyc`,
  `.DS_Store`, `.worktrees/`.
- ✓ `uv` available at `~/.local/bin/uv`.
- ✗ Host Python is 3.9.7 (anaconda); design assumes 3.11. **Resolution:** Use
  `uv venv --python 3.11` to fetch and pin 3.11 inside the venv. No host
  Python upgrade required.
- ✓ `ffmpeg 4.2.2` on PATH (needed in Phase 3, verified here).
- ✗ `yt-dlp` not on PATH — fine, it is a pip dependency added in Phase 3.
- ✓ No prior tests/, pyproject.toml, requirements.txt, conftest.py — Phase 1
  authors them fresh.
- ✓ No worktree-local CLAUDE.md or AGENTS.md.

---

## Acceptance Criteria Coverage

This phase is infrastructure-only. **Verifies: None** (no functional ACs).
All ACs are covered in later phases.

The deliverables of this phase are gated by **operational** verification:
- Fresh `./setup.sh` succeeds.
- `python -c "import config; import pipeline.types; import pipeline.storage; import providers.llm; import providers.vision; import providers.embed; import pricing"` exits 0.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Project skeleton — `pyproject.toml`, `requirements.txt`, lockfile

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Verify: `.gitignore` already contains expected entries (no change expected).

**Implementation:**

`pyproject.toml` declares project metadata, Python ≥3.11, and a minimal `[tool.pytest.ini_options]` section so pytest discovers tests under `tests/`. Use the PEP 621 layout. Do NOT use Hatch/Poetry build backends — keep it simple; project is not published to PyPI.

```toml
[project]
name = "video-to-steps"
version = "0.1.0"
description = "YouTube instructional video → ordered illustrated step-by-step guide."
requires-python = ">=3.11"
readme = "README.md"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "cloud: tests that call cloud APIs; skipped unless RUN_CLOUD_TESTS=1",
]
```

`requirements.txt` (Phase 1 needs only the config-layer deps; downstream phases append):

```
pydantic>=2.7
pydantic-settings>=2.3
```

`requirements-dev.txt`:

```
-r requirements.txt
pytest>=8.0
pytest-asyncio>=0.23
```

**Verification:**

```bash
ls pyproject.toml requirements.txt requirements-dev.txt
```
Expected: all three present.

**Commit:**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt
git commit -m "chore(vts-v1): bootstrap pyproject + requirements"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `.env.example` with all three modes

**Files:**
- Create: `.env.example`

**Implementation:**

Document **every** environment variable the application reads, grouped by
mode. Mode C is uncommented (it's the v1 acceptance default); Mode A and B
blocks are commented with `#`.

The exact variable names below MUST match what `config.Settings` reads in
Task 4 — they are the contract between operator and code.

```bash
# .env.example — copy to .env and fill in API keys before running.
# Three deployment modes share the same code; switch via env only.

# ============================================================================
# Server
# ============================================================================
APP_HOST=127.0.0.1          # set to 0.0.0.0 for cloud deploy (behind a reverse proxy)
APP_PORT=8090
JOBS_ROOT=./data/jobs       # per-job artifacts live here

# ============================================================================
# Mode C — Cloud (default; v1 acceptance target)
# ============================================================================
EMBED_BACKEND=jina_v4
JINA_API_KEY=                       # required for Mode C
JINA_MODEL=jina-embeddings-v4
JINA_EMBED_BATCH=64

LLM_BASE_URL=https://api.deepseek.com
LLM_PATH_CHAT=/v1/chat/completions
LLM_API_KEY=                        # required for Mode C
LLM_MODEL=deepseek-chat
LLM_MAX_TOKENS=2048
LLM_INCLUDE_USAGE=1                 # set to 0 for providers that 400 on stream_options

VISION_BASE_URL=https://api.openai.com
VISION_PATH_CHAT=/v1/chat/completions
VISION_API_KEY=                     # required for Mode C
VISION_MODEL=gpt-4o-mini
VISION_MAX_TOKENS=300
VISION_INCLUDE_USAGE=1              # set to 0 for providers that 400 on stream_options

REFINE_MAX_IN_FLIGHT=4
CAPTION_MAX_IN_FLIGHT=16
WHISPER_FALLBACK=0                  # v2 roadmap

# ============================================================================
# Mode A — Local (Macbook + qwen-studio + mlx_clip). UNTESTED in v1.
# Uncomment this block (and comment Mode C above) to try Mode A.
# ============================================================================
# EMBED_BACKEND=mlx_clip
# MLX_CLIP_MODEL=openai/clip-vit-base-patch32
#
# LLM_BASE_URL=http://127.0.0.1:8766
# LLM_PATH_CHAT=/chat
# LLM_API_KEY=                       # qwen-studio ignores; can be blank
# LLM_MODEL=qwen2.5-coder            # whatever qwen-studio is serving
# LLM_MAX_TOKENS=2048
# LLM_INCLUDE_USAGE=0                # qwen-studio is raw-text SSE; no usage chunk
#
# VISION_BASE_URL=http://127.0.0.1:8766
# VISION_PATH_CHAT=/chat
# VISION_API_KEY=
# VISION_MODEL=qwen-vl
# VISION_INCLUDE_USAGE=0

# ============================================================================
# Mode B — Hybrid (MLX CLIP locally + cloud LLM + cloud vision). UNTESTED in v1.
# Same as Mode C but with EMBED_BACKEND=mlx_clip.
# ============================================================================
```

**Verification:**

```bash
test -f .env.example && grep -q "^APP_HOST=127.0.0.1" .env.example && echo OK
```
Expected: `OK`.

**Commit:**

```bash
git add .env.example
git commit -m "chore(vts-v1): add .env.example covering all three modes"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `setup.sh`, `start.sh`, `stop.sh`

**Files:**
- Create: `setup.sh` (executable)
- Create: `start.sh` (executable)
- Create: `stop.sh` (executable)

**Implementation:**

Plain bash scripts (no fancy features) so a fresh reader can audit them in
30 seconds. `setup.sh` provisions a `uv`-managed Python 3.11 venv; `start.sh`
binds host/port from env; `stop.sh` kills the foreground process by PID file.

`setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv venv --python 3.11 venv
# shellcheck source=/dev/null
source venv/bin/activate
uv pip install -r requirements-dev.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not on PATH. Install before running pipeline." >&2
fi

echo "Setup complete. Activate venv with: source venv/bin/activate"
```

`start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source venv/bin/activate

# Load .env if present (don't fail if missing — env vars may come from elsewhere).
if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

mkdir -p "${JOBS_ROOT:-./data/jobs}"

HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8090}"

# server.py is created in Phase 6; until then this will exit non-zero. That's fine.
python -m uvicorn server:app --host "$HOST" --port "$PORT" &
echo $! > .vts.pid
echo "Started on $HOST:$PORT (pid $(cat .vts.pid))"
```

`stop.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -f .vts.pid ]; then
  PID=$(cat .vts.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped pid $PID"
  else
    echo "Process $PID not running"
  fi
  rm -f .vts.pid
else
  echo ".vts.pid not found; nothing to stop"
fi
```

Make all three executable.

**Verification:**

```bash
chmod +x setup.sh start.sh stop.sh
ls -l setup.sh start.sh stop.sh | grep -c '^-rwx' | grep -q 3 && echo OK
bash -n setup.sh && bash -n start.sh && bash -n stop.sh && echo "syntax OK"
./setup.sh
test -d venv && echo "venv OK"
```
Expected: `OK` and `syntax OK` and `venv OK`; setup completes without
errors (apart from possibly an `ffmpeg` warning — acceptable).

**Commit:**

```bash
git add setup.sh start.sh stop.sh
git commit -m "chore(vts-v1): add setup/start/stop scripts"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: `config.py` — Pydantic `Settings`

**Files:**
- Create: `config.py`

**Implementation:**

Single class `Settings(BaseSettings)` reading `.env` plus process env, with
defaults that make Mode C the runtime default. Field names map 1:1 to the
env-var names in `.env.example`. Every secret defaults to empty string so
import-time doesn't fail when `.env` is absent; downstream factories raise
clear errors when a required key is empty.

```python
"""Runtime configuration.

Single source of truth for env-driven settings. Imported everywhere; never
mutated at runtime.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8090, alias="APP_PORT")
    jobs_root: Path = Field(default=Path("./data/jobs"), alias="JOBS_ROOT")

    # Embedder
    embed_backend: str = Field(default="jina_v4", alias="EMBED_BACKEND")
    jina_api_key: str = Field(default="", alias="JINA_API_KEY")
    jina_model: str = Field(default="jina-embeddings-v4", alias="JINA_MODEL")
    jina_embed_batch: int = Field(default=64, alias="JINA_EMBED_BATCH")
    mlx_clip_model: str = Field(default="openai/clip-vit-base-patch32", alias="MLX_CLIP_MODEL")

    # Text LLM
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_path_chat: str = Field(default="/v1/chat/completions", alias="LLM_PATH_CHAT")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    # Some OpenAI-strict providers (gpt-*, DeepSeek, Together) accept
    # stream_options={"include_usage": true} for token counts in the final
    # SSE chunk; others (qwen-studio, some local proxies) may 400 on
    # unknown top-level params. Default True; flip to False for strict
    # providers that reject it.
    llm_include_usage: bool = Field(default=True, alias="LLM_INCLUDE_USAGE")

    # Vision LLM
    vision_base_url: str = Field(default="https://api.openai.com", alias="VISION_BASE_URL")
    vision_path_chat: str = Field(default="/v1/chat/completions", alias="VISION_PATH_CHAT")
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")
    vision_model: str = Field(default="gpt-4o-mini", alias="VISION_MODEL")
    vision_max_tokens: int = Field(default=300, alias="VISION_MAX_TOKENS")
    vision_include_usage: bool = Field(default=True, alias="VISION_INCLUDE_USAGE")

    # Concurrency
    refine_max_in_flight: int = Field(default=4, alias="REFINE_MAX_IN_FLIGHT")
    caption_max_in_flight: int = Field(default=16, alias="CAPTION_MAX_IN_FLIGHT")

    # Feature flags
    whisper_fallback: bool = Field(default=False, alias="WHISPER_FALLBACK")


def get_settings() -> Settings:
    """Returns a fresh Settings instance. Callers may cache as needed."""
    return Settings()
```

**Verification:**

```bash
source venv/bin/activate
python -c "from config import get_settings; s = get_settings(); print(s.app_host, s.app_port, s.embed_backend)"
```
Expected output: `127.0.0.1 8090 jina_v4`.

**Commit:**

```bash
git add config.py
git commit -m "feat(vts-v1): add Pydantic Settings for all three modes"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `pipeline/types.py` + `pipeline/storage.py`

**Files:**
- Create: `pipeline/__init__.py` (empty)
- Create: `pipeline/types.py`
- Create: `pipeline/storage.py`

**Implementation:**

Plain `@dataclass` definitions for internal pipeline data. Use `slots=True`
so type errors surface early. `Manifest` is the on-disk record served to the
status-poll endpoint; its JSON shape MUST be stable from this phase forward.

`pipeline/types.py`:

```python
"""Core pipeline data types.

These are plain dataclasses (not Pydantic) — they're internal contracts, not
external boundaries. Pydantic is reserved for env-driven Settings and any
future incoming HTTP payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Cue:
    """A single timed caption segment parsed from a VTT track."""
    start: float          # seconds
    end: float            # seconds
    text: str


@dataclass(slots=True)
class Frame:
    """One extracted still frame, located in time."""
    index: int            # 0-based ordinal within the frame set
    timestamp: float      # seconds since start of video
    path: Path            # absolute path to the .jpg


@dataclass(slots=True)
class StepOutline:
    """LLM Pass 1 output — coarse step boundary + brief description."""
    index: int            # 0-based ordinal
    start: float          # seconds
    end: float            # seconds
    brief: str            # ≤ 1 sentence describing the step


@dataclass(slots=True)
class Step:
    """LLM Pass 2 output — polished step text + selected illustrating frames."""
    index: int
    start: float
    end: float
    instruction: str      # 1–3 second-person imperative sentences
    frames: list[Frame] = field(default_factory=list)


@dataclass(slots=True)
class TokenUsage:
    """Billable-token counts for a single provider call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0


@dataclass(slots=True)
class CostBreakdown:
    """Running cost totals for a job, in USD."""
    chat_usd: float = 0.0
    vision_usd: float = 0.0
    embed_usd: float = 0.0
    total_usd: float = 0.0


@dataclass(slots=True)
class Manifest:
    """Per-job record persisted to meta.json.

    Only the orchestrator mutates this; the server reads from disk.
    """
    job_id: str
    url: str
    status: str = "queued"             # queued | running | done | error
    progress: str = ""                 # free-form short description of current stage
    error: str = ""                    # populated when status == "error"
    mode: str = ""                     # "cloud" | "local" | "hybrid" — informational
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
```

`pipeline/storage.py`:

```python
"""Atomic JSON write helpers and job-directory resolver.

`write_json_atomic` writes to `path.tmp` then `os.replace(tmp, path)` — on
POSIX this is a single rename inode operation, so concurrent readers (the
HTMX status poll) see either the old or new file but never a torn write.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def job_dir(jobs_root: Path, job_id: str) -> Path:
    """Returns the per-job artifact directory; does NOT create it."""
    return Path(jobs_root) / job_id


def ensure_job_dir(jobs_root: Path, job_id: str) -> Path:
    """Returns the per-job artifact directory, creating it if absent."""
    d = job_dir(jobs_root, job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)
    return d


def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, Paths, and sets to JSON-safe shapes."""
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value)]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    """Write `value` as JSON to `path` atomically.

    Writes to `path.with_suffix(path.suffix + ".tmp")` in the same directory,
    then atomically renames. The renaming guarantees readers never see a
    half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = _to_jsonable(value)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Read JSON from `path`. Raises FileNotFoundError if absent."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
```

**Verification:**

```bash
source venv/bin/activate
python -c "
from pathlib import Path
from pipeline.types import Manifest, CostBreakdown
from pipeline.storage import write_json_atomic, read_json, ensure_job_dir
import tempfile

m = Manifest(job_id='abc123', url='https://youtu.be/x', cost=CostBreakdown(total_usd=0.12))
with tempfile.TemporaryDirectory() as td:
    d = ensure_job_dir(Path(td), 'abc123')
    write_json_atomic(d / 'meta.json', m)
    loaded = read_json(d / 'meta.json')
    assert loaded['job_id'] == 'abc123'
    assert loaded['cost']['total_usd'] == 0.12
print('OK')
"
```
Expected: `OK`.

**Commit:**

```bash
git add pipeline/
git commit -m "feat(vts-v1): pipeline types + atomic JSON storage"
```
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 6-7) -->

<!-- START_TASK_6 -->
### Task 6: `pricing.py` — per-model price table

**Files:**
- Create: `pricing.py`

**Implementation:**

Static dict keyed by `model_id`. Missing models are tolerated: lookup returns
zeros and a startup warning fires once per process via `_warn_once`. The
orchestrator calls `compute_chat_cost`, `compute_vision_cost`,
`compute_embed_cost` once per stage and accumulates into `CostBreakdown`.

Initial entries cover the v1 cloud defaults; readers can append rows without
touching call sites. Prices are USD per 1,000,000 tokens.

```python
"""Per-model pricing for cost reporting.

Edit `PRICES` to add new models. Missing models record zeros and log a
warning at startup; the pipeline still completes.

All prices are USD per 1,000,000 tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    prompt_per_million: float
    completion_per_million: float
    embed_per_million: float = 0.0


# Last reviewed: 2026-05. Spot-check against provider pricing pages before
# trusting absolute numbers for budgeting; the goal here is rough cost
# visibility per run, not finance-grade accounting.
PRICES: dict[str, ModelPrice] = {
    # Text LLM
    "deepseek-chat":      ModelPrice(prompt_per_million=0.27, completion_per_million=1.10),
    "gpt-4o-mini":        ModelPrice(prompt_per_million=0.15, completion_per_million=0.60),
    "gpt-4o":             ModelPrice(prompt_per_million=2.50, completion_per_million=10.00),
    # Vision
    # (gpt-4o-mini is dual-use; same price table entry as above)
    # Embeddings — Jina charges per token; image tokens depend on tile count.
    "jina-embeddings-v4": ModelPrice(prompt_per_million=0.0, completion_per_million=0.0, embed_per_million=0.18),
}


_warned: set[str] = set()


def _warn_once(model_id: str) -> None:
    if model_id not in _warned:
        _warned.add(model_id)
        log.warning("pricing.py: no entry for model %r; cost will record zero.", model_id)


def _price_or_zero(model_id: str) -> ModelPrice:
    p = PRICES.get(model_id)
    if p is None:
        _warn_once(model_id)
        return ModelPrice(0.0, 0.0, 0.0)
    return p


def compute_chat_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = _price_or_zero(model_id)
    return (prompt_tokens / 1_000_000.0) * p.prompt_per_million \
         + (completion_tokens / 1_000_000.0) * p.completion_per_million


def compute_vision_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Vision is just a chat call with image parts; same pricing shape.
    return compute_chat_cost(model_id, prompt_tokens, completion_tokens)


def compute_embed_cost(model_id: str, total_tokens: int) -> float:
    p = _price_or_zero(model_id)
    return (total_tokens / 1_000_000.0) * p.embed_per_million
```

**Verification:**

```bash
source venv/bin/activate
python -c "
from pricing import compute_chat_cost, compute_embed_cost
# Known model
assert compute_chat_cost('deepseek-chat', 1_000_000, 500_000) == 0.27 + (0.5 * 1.10)
# Unknown model returns zero (and logs warning)
import logging; logging.basicConfig()
assert compute_chat_cost('unknown-model-xyz', 1_000_000, 1_000_000) == 0.0
print('OK')
"
```
Expected: `OK` and a `WARNING:pricing:... no entry for model 'unknown-model-xyz'` line.

**Commit:**

```bash
git add pricing.py
git commit -m "feat(vts-v1): static price table with safe-zero fallback"
```
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: `providers/` — protocols and factory stubs

**Files:**
- Create: `providers/__init__.py` (empty)
- Create: `providers/llm.py`
- Create: `providers/vision.py`
- Create: `providers/embed.py`

**Implementation:**

Protocols only. Each file ships a `Protocol` class and a `build_X(settings)`
factory that raises `NotImplementedError` for now. Phase 2 replaces the
stubs with concrete classes; Phase 1's job is to nail the contracts so Phase
2 can fan out work without bikeshedding.

`providers/llm.py`:

```python
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
```

`providers/vision.py`:

```python
"""VisionCaptioner protocol + factory stub.

Concrete implementation lands in Phase 2. Caption-of-winners only; this is
NOT used to caption every frame. See design § Architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class CaptionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class VisionCaptioner(Protocol):
    name: str

    async def caption(self, image: Path) -> CaptionResult: ...


def build_vision(settings: Any) -> VisionCaptioner:
    """Returns a configured VisionCaptioner. Implemented in Phase 2."""
    raise NotImplementedError("VisionCaptioner factory implemented in Phase 2")
```

`providers/embed.py`:

```python
"""Embedder + FrameExtractor protocols, embedder factory stub.

Embedder vectors MUST be float32, shape (n, d), L2-normalized. The protocol
makes this explicit so cosine similarity reduces to a plain `frame_emb @
step_emb`.
"""

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
        """Returns a list[pipeline.types.Frame]. Implementations in Phase 3."""
        ...


def build_embedder(settings: Any) -> Embedder:
    """Returns a configured Embedder. Implemented in Phase 2."""
    raise NotImplementedError("Embedder factory implemented in Phase 2")
```

Add `numpy` to `requirements.txt` since `providers/embed.py` imports it (Phase 2 needs it anyway, but the import must work today for the Phase 1 verification command). Append:

```
numpy>=1.26
```

Re-run `uv pip install -r requirements-dev.txt` after editing.

**Verification:**

```bash
source venv/bin/activate
uv pip install -r requirements-dev.txt
python -c "
import providers.llm, providers.vision, providers.embed
# Factories raise NotImplementedError until Phase 2
try:
    providers.llm.build_llm(None)
    assert False, 'expected NotImplementedError'
except NotImplementedError:
    pass
try:
    providers.embed.build_embedder(None)
except NotImplementedError:
    pass
print('OK')
"
```
Expected: `OK`.

**Commit:**

```bash
git add requirements.txt providers/
git commit -m "feat(vts-v1): provider protocols (LLMClient, VisionCaptioner, Embedder, FrameExtractor)"
```
<!-- END_TASK_7 -->

<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_8 -->
### Task 8: Operational verification + final commit

**Files:** None created. This task verifies Phase 1 as a whole.

**Verification:**

```bash
# Wipe and rebuild venv to prove setup.sh works from scratch.
rm -rf venv
./setup.sh
source venv/bin/activate

# Import sanity — exactly the command from the design plan's "Done when".
python -c "import config; import pipeline.types; import pipeline.storage; import providers.llm; import providers.vision; import providers.embed; import pricing"
echo "imports OK"
```
Expected: scripts run cleanly, no traceback, `imports OK` printed.

Nothing to commit (no file changes); this task is a gate, not a deliverable.

**Done when:** Phase 1's "Done when" from the design plan is satisfied:
1. `./setup.sh` succeeds in a fresh venv.
2. The 7-module import command runs cleanly.
<!-- END_TASK_8 -->
