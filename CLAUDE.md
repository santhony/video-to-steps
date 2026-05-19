# video-to-steps

Last verified: 2026-05-19

YouTube instructional video → ordered illustrated step-by-step guide. v1
implementation lives on the `vts-v1` branch; design plan at
`docs/design-plans/2026-05-18-vts-v1.md`, phase-by-phase implementation
plans at `docs/implementation-plans/2026-05-18-vts-v1/`.

## Tech Stack
- Python 3.11+ (pinned in `pyproject.toml`)
- FastAPI + Jinja2 + HTMX (server, vendored htmx at `static/js/htmx.min.js`)
- `httpx` (async, all provider HTTP)
- `pydantic-settings` for env-driven config (Pydantic itself reserved for
  config and any future incoming HTTP payloads — internal data flow is
  plain `@dataclass(slots=True)`, not Pydantic models)
- `yt-dlp` + `webvtt-python` + `imagehash` + `ffmpeg` (Phase 3 ingest)
- `numpy` (embeddings, matching)
- Jina v4 embeddings (cloud, default); `mlx_clip` (untested local path)
- Provider-agnostic OpenAI-shape chat + vision endpoints

## Commands
- `./setup.sh` — create `venv/` and install deps (uses `uv` to fetch
  Python 3.11 if the host has older).
- `./start.sh` — launch uvicorn (PID in `.vts.pid` in repo root; no log
  file is written — uvicorn output goes to stdout of the parent shell).
- `./stop.sh` — graceful shutdown via the PID file.
- `python -m pytest` — runs the offline test suite (cloud tests are
  marked `@pytest.mark.cloud` and skipped unless `RUN_CLOUD_TESTS=1`).
- `python -m scripts.smoke_llm` / `smoke_vision` / `smoke_embed` /
  `smoke_phase3` — one-call provider diagnostics. Each prints first
  ~100 chars + token usage; use to verify a fresh `.env` end-to-end.

## Project Structure
- `config.py` — `Settings` (pydantic-settings, all fields `Field(alias=...)`
  matching `.env.example` exactly). `get_settings()` returns a fresh
  instance per call; callers may cache.
- `pricing.py` — static `PRICES` table; missing models record zero and
  warn-once.
- `providers/` — adapters to external services. One module per provider
  family. Public protocols live in `embed.py`; concrete LLM/vision
  clients in `llm.py` / `vision.py`. Jina + mlx_clip embedders are
  separate files behind a factory in `embed.py`.
- `pipeline/` — all video→steps stages. Pure logic in `captions.py`,
  `match.py`, parsing helpers; orchestration in `pipeline.py`.
- `prompts/` — markdown templates: `outline.md`, `refine.md`,
  `vision_caption.md`. The first two split on a `## User` heading via
  `pipeline._prompts.load_system_user`.
- `server.py` — FastAPI app (form, process, job page, status fragment,
  result page, frame).
- `templates/`, `static/` — Jinja2 templates + vendored htmx/css.
- `scripts/` — one-call smoke diagnostics, not part of runtime.
- `tests/` — mirrors `pipeline/` and `providers/` layout.

## Conventions

### Functional Core / Imperative Shell (FCIS)
Every non-trivial module declares its half at the top of its docstring:
- **Functional Core**: `pipeline/match.py`, plus pure helpers inside
  `pipeline/captions.py`, `pipeline/frames.py`, `pipeline/llm_outline.py`,
  `pipeline/llm_refine.py`.
- **Imperative Shell**: `pipeline/pipeline.py`, `pipeline/caption_winners.py`,
  `pipeline/_prompts.py`, `pipeline/captions.py`, `pipeline/download.py`,
  `pipeline/frames.py`, `pipeline/storage.py`, all files in `providers/`,
  and `server.py`.

Pure logic stays free of I/O; the shell composes it with HTTP, disk, and
ffmpeg.

### Pydantic vs dataclass
- **Pydantic**: only `config.Settings` (env boundary). Reserve for any
  future incoming HTTP payload validation.
- **`@dataclass(slots=True)`**: every internal data type (`Cue`,
  `Frame`, `StepOutline`, `Step`, `TokenUsage`, `CostBreakdown`,
  `Manifest`, `ChatResult`, `CaptionResult`, `EmbedResult`, `ModelPrice`).
  `slots=True` is intentional — do not drop it; it catches typos.

### Atomic JSON writes
Anything written to `data/jobs/<id>/*.json` MUST go through
`pipeline.storage.write_json_atomic`. It writes to `<path>.tmp` +
`os.replace` so the HTMX status poll never sees a torn file.
`_to_jsonable` recursively unwraps dataclasses, `Path`, sets, and nested
dicts/lists — pass dataclasses directly, no manual `asdict`.

### Test markers
`@pytest.mark.cloud` for tests that hit live providers. Default
`addopts = -m 'not cloud' --strict-markers` keeps the offline suite
fast; `RUN_CLOUD_TESTS=1 pytest -m cloud` runs them.

## Provider Protocols (Contracts)

All four providers expose a `name: str` attribute (NOT a method — a
plain attribute set in `__init__`). The orchestrator uses it for pricing
lookup; never call `.name()`.

- **`providers.llm.LLMClient`** — `async chat(messages, *, max_tokens,
  response_format) → ChatResult(text, prompt_tokens, completion_tokens)`.
  Auto-detects OpenAI-vs-qwen-studio SSE shape by peeking the first
  `data:` payload. Strips `<think>...</think>` unconditionally; ignores
  `reasoning_content` (DeepSeek v4 CoT goes there — if `LLM_MAX_TOKENS`
  is too low the content field returns empty, raise the cap to ≥2048).
- **`providers.vision.VisionCaptioner`** — `async caption(image: Path)
  → CaptionResult`. OpenAI-shape only (no qwen-studio raw-text path);
  per-frame failures bubble up as exceptions and `caption_winners`
  catches them.
- **`providers.embed.Embedder` (Protocol)** — `async embed_images`,
  `async embed_texts` → `EmbedResult(vectors, billable_tokens)`.
  Vectors MUST be float32, shape `(n, d)`, L2-normalized. `match()`
  assumes `frame_emb @ step_emb.T` IS cosine similarity.
  `JinaEmbedder` defensively re-normalizes; `MlxClipEmbedder` is
  import-guarded and untested in v1.
- **`providers.embed.FrameExtractor` (Protocol)** — `extract(video,
  out_dir) → list[Frame]`. `FixedFpsExtractor` (ffmpeg + pHash dedup)
  is the only implementation used in v1; `SceneChangeExtractor` is a v2
  stub.

The factories `build_llm`, `build_vision`, `build_embedder` are the
only sanctioned construction sites — the orchestrator calls them and
never instantiates clients directly.

## The 10-Stage Pipeline

`pipeline.pipeline.run_job(job_id, url, settings, jobs_root)` is the
single orchestrator. Stages (each preceded by a `_update(progress=...)`
manifest write):
1. Download video + VTT (`pipeline.download.download_video_and_captions`).
   No captions → manifest error "This video has no captions" and early
   return (NOT raise), unless `whisper_fallback=True` (v2 roadmap).
2. Parse + dedupe captions (`parse_vtt` → `dedupe_rolling`).
3. Extract frames @ 1 fps with pHash dedup (`FixedFpsExtractor`,
   `hamming_max=6`).
4. Embed every frame (`embedder.embed_images`); save
   `frame_embeddings.npy`.
5. LLM Pass 1 outline → `outline.json` (`llm_outline`, json_object +
   slice-fallback).
6. Embed step briefs (`embedder.embed_texts`).
7. Match frames to steps (`match`, pure; top-k=3, pad_sec=2,
   midpoint-nearest fallback for empty windows).
8. Caption only the union of winner frames (`caption_winners`,
   semaphore-bounded fanout, per-frame failure tolerated).
9. LLM Pass 2 refine each step (`llm_refine`, semaphore fanout, slice
   fallback, degraded path drops captions if refine fails).
10. Persist `steps.json` and accumulate `CostBreakdown` from
    `pricing.compute_*`.

Manifest lifecycle: `queued` (server, in `POST /process`) → `running`
(stage 0) → stage labels → `done` or `error`. The orchestrator NEVER
sets `queued`. The outer `try/except` sets `error` AND re-raises
(server's `BackgroundTasks` swallows the re-raise; the manifest is the
source of truth).

`pipeline._prompts.load_system_user` is the shared prompt loader for
the two text-LLM passes; vision uses its own `prompts/vision_caption.md`
loaded inline in `providers/vision.py`.

`pipeline.types.TokenUsage` is the canonical chat usage shape returned
by `llm_outline`, `llm_refine`, and `caption_winners` (NOT
`ChatResult`/`CaptionResult` — those are provider-level).

## Server Routes
- `GET /` → form (`index.html`).
- `POST /process` → validate YouTube URL via `_YT_RE`, allocate 12-hex
  job_id, write initial queued manifest, spawn `run_job` via
  `BackgroundTasks`, 303 to `/job/{id}`.
- `GET /job/{id}` → poll page (`job.html`).
- `GET /job/{id}/status` → HTMX fragment, served as either
  `status_fragment.html` (still running) or `status_done_fragment.html`
  (terminal). Templates use `_AttrDict`-wrapped dicts for attribute
  syntax.
- `GET /job/{id}/result` → result page; 303 back to job page if not
  yet `done`.
- `GET /job/{id}/frame/{name}.jpg` → serve `data/jobs/<id>/frames/{name}.jpg`.

Every route validates `job_id` via `_JOB_ID_RE = ^[a-f0-9]{12}$`. The
frame route additionally requires `name` to be a 4-digit string. Bad
input → 400, missing job → 404.

## Settings (Env Boundary)

All fields use `Field(alias="UPPER_SNAKE")` matching `.env.example`
exactly. Three deployment modes (see README for full env tables):
- **Mode C — Cloud** (default; the ONLY mode tested in v1): Jina v4 +
  cloud LLM + cloud vision.
- **Mode A — Local Mac**: mlx_clip + qwen-studio. `MlxClipEmbedder` is
  import-guarded; raises `RuntimeError` on non-Apple-Silicon.
  **Untested in v1.**
- **Mode B — Hybrid**: mlx_clip + cloud LLM + cloud vision. No separate
  code path; just Mode C with `EMBED_BACKEND=mlx_clip`. **Untested.**

`llm_include_usage` and `vision_include_usage` opt out of
`stream_options.include_usage` for strict providers (qwen-studio) that
400 on unknown top-level params.

## Job Directory Layout

```
data/jobs/<job_id>/
  meta.json              ← Manifest (atomic write, single writer)
  video.mp4              ← yt-dlp output
  *.en.vtt               ← yt-dlp captions
  frames/0000.jpg, …     ← FixedFpsExtractor (4-digit zero-pad)
  frame_embeddings.npy   ← embedder.embed_images output
  outline.json           ← list[StepOutline]
  frame_captions.json    ← {frame_index: caption|null}
  steps.json             ← list[Step] (final result)
```

The server reads this directory directly — there is no in-memory job
store. Manifest is the only source of truth for status.

## Key Decisions
- **Caption only winners, not every frame** (cost): caption pass runs
  on the ~30 union-of-winners, not the full ~180 frames of a 3-min
  video. Each winner is captioned exactly once even if it wins for
  multiple steps.
- **Atomic JSON over locks**: single-writer orchestrator + atomic
  rename means the HTMX poll never reads a torn file without any
  locking primitive.
- **Plain dataclass over Pydantic for internal types**: zero overhead,
  `slots=True` catches typos, no boundary that needs validation.
- **OpenAI-shape SSE for everything chat-like**: one parser, both
  providers and qwen-studio's raw-text shape are auto-detected by the
  same client.
- **Static price table over provider rate API**: rates change rarely;
  hand-edit `pricing.py` and document the review date in the
  `# Last reviewed:` comment.

## Invariants
- Embedder vectors are float32, shape `(n, d)`, L2-normalized at exit
  of `embed_images` / `embed_texts`. `match()` depends on this.
- Provider clients expose `name` as an attribute, never a method.
- Internal data types use `@dataclass(slots=True)` — do not remove
  `slots`; do not introduce Pydantic for internal types.
- All `data/jobs/<id>/*.json` writes go through `write_json_atomic`.
- Manifest writes only happen inside `pipeline.pipeline._update` (in
  the orchestrator) or directly in `server.py` for the initial
  `queued` state. Two writers, same atomic helper.
- Job IDs match `^[a-f0-9]{12}$`. Frame filenames are 4-digit zero-pad.
- `pipeline._prompts.load_system_user` requires exactly one `## User`
  heading per prompt file; it raises `RuntimeError` otherwise.

## Gotchas
- **DeepSeek v4 empty content**: reasoning models emit CoT in
  `reasoning_content` (which we ignore). If `LLM_MAX_TOKENS` is too
  low the visible `content` comes back empty. Raise to ≥2048.
- **qwen-studio SSE shape**: each `data:` line is the literal next-token
  text, not a JSON object. `LLMClient` auto-detects this. Some strict
  providers 400 on `stream_options` — see `*_INCLUDE_USAGE` flags.
- **Pricing zero**: missing model entries in `pricing.py` record zero
  and warn-once. Recorded cost will be a lower bound; check the log.
- **Caption tolerance**: a vision provider that refuses an image
  (people / content policy) is recorded as `caption=None` and the
  step refine pass uses cue text only. The manifest still ends `done`.
- **Mode A / Mode B untested**: factory routes correctly and the
  embedder is import-guarded, but no v1 smoke run has gone through
  these paths. Treat as "wired but not battle-tested."
- **Pending operator items**: README rehearsal at
  `docs/implementation-plans/2026-05-18-vts-v1/rehearsal-notes.md`
  lists fresh-host steps that a CI-like agent cannot verify
  (`setup.sh` / `start.sh` / Docker build / real YouTube run).

## Boundaries
- Safe to edit: `pipeline/`, `providers/`, `server.py`, `templates/`,
  `static/css/main.css`, `prompts/`, `scripts/`, `tests/`, `config.py`,
  `pricing.py`, `README.md`.
- Treat as vendored / generated: `static/js/htmx.min.js`,
  `venv/`, `__pycache__/`, `data/`, `.ruff_cache/`, `.pytest_cache/`.
- Do not touch without a design decision: the FCIS pattern markers in
  module docstrings (they are part of the discipline, not decoration);
  `Field(alias=...)` on `Settings` (the env var names are the public
  contract for operators); `slots=True` on dataclasses; the atomic
  write helper.
