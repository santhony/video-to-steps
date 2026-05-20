# video-to-steps

Last verified: 2026-05-19

YouTube instructional video → ordered illustrated step-by-step guide. v1
implementation lives on the `vts-v1` branch; design plan at
`docs/design-plans/2026-05-18-vts-v1.md`, phase-by-phase implementation
plans at `docs/implementation-plans/2026-05-18-vts-v1/`. v1.1 adds the
Whisper transcription fallback (no longer a v2 stub).

## Tech Stack
- Python 3.11+ (pinned in `pyproject.toml`)
- FastAPI + Jinja2 + HTMX (server, vendored htmx at `static/js/htmx.min.js`)
- `httpx` (async, all provider HTTP)
- `pydantic-settings` for env-driven config (Pydantic itself reserved for
  config and any future incoming HTTP payloads — internal data flow is
  plain `@dataclass(slots=True)`, not Pydantic models)
- `yt-dlp` + `webvtt-python` + `imagehash` + `ffmpeg` (Phase 3 ingest;
  `ffmpeg` is also used for audio extraction in the Whisper fallback)
- `Pillow` (frame thumbnailing for embedding)
- `numpy` (embeddings, matching)
- Jina v4 embeddings (cloud, default); `mlx_clip` (untested local path)
- `faster-whisper` (optional, lazy-imported; only loaded when
  `WHISPER_FALLBACK=1` and the video has no YouTube captions)
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
  family. Public protocols live in `embed.py` and `whisper.py`;
  concrete LLM/vision clients in `llm.py` / `vision.py`. Jina + mlx_clip
  embedders are separate files behind a factory in `embed.py`. The
  Whisper transcriber (`whisper.py`) is its own module behind a
  `build_whisper(settings)` factory.
- `pipeline/` — all video→steps stages. Pure logic in `captions.py`,
  `match.py`, parsing helpers; orchestration in `pipeline.py`. Audio
  extraction for the Whisper fallback lives in `audio.py`.
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
  `pipeline/_prompts.py`, `pipeline/audio.py`, `pipeline/captions.py`,
  `pipeline/download.py`, `pipeline/frames.py`, `pipeline/storage.py`,
  all files in `providers/`, and `server.py`.

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

All providers expose a `name: str` attribute (NOT a method — a plain
attribute set in `__init__`). The orchestrator uses it for pricing
lookup; never call `.name()`.

- **`providers.llm.LLMClient`** — `async chat(messages, *, max_tokens,
  response_format) → ChatResult(text, prompt_tokens, completion_tokens)`.
  Auto-detects OpenAI-vs-qwen-studio SSE shape by peeking the first
  `data:` payload. Strips `<think>...</think>` unconditionally; ignores
  `reasoning_content` (DeepSeek v4 CoT goes there — if max_tokens is too
  low the content field returns empty). `llm_refine` hard-codes
  `max_tokens=1500` for this reason (covers ~1200 reasoning + ~300
  visible content for a refined step); `llm_outline` uses
  `settings.llm_max_tokens` for the longer outline pass. The refine
  system prompt targets 1–3 second-person imperative sentences per
  step with a specificity directive to keep concrete tool names,
  quantities, direction, and timing rather than summarizing them away.
  It also includes an explicit voice rule (with examples) that
  forbids third-person descriptions of the on-screen person
  ("the host", "the instructor", "she", etc.) — local models tend to
  default to narrating the speaker without it.
- **`providers.vision.VisionCaptioner`** — `async caption(image: Path)
  → CaptionResult`. OpenAI-shape only (no qwen-studio raw-text path);
  per-frame failures bubble up as exceptions and `caption_winners`
  catches them.
- **`providers.embed.Embedder` (Protocol)** — `async embed_images`,
  `async embed_texts` → `EmbedResult(vectors, billable_tokens)`.
  Vectors MUST be float32, shape `(n, d)`, L2-normalized. `match()`
  assumes `frame_emb @ step_emb.T` IS cosine similarity.
  `JinaEmbedder` defensively re-normalizes AND retries up to 5x on
  HTTP 429 (honors `Retry-After` header, else exponential backoff
  capped at 60s); non-429 errors raise immediately. Callers must
  tolerate longer-than-expected latency when the free tier throttles.
  `MlxClipEmbedder` is import-guarded and untested in v1.
- **`providers.embed.FrameExtractor` (Protocol)** — `extract(video,
  out_dir) → list[Frame]`. `FixedFpsExtractor` (ffmpeg + pHash dedup)
  is the only implementation used in v1; `SceneChangeExtractor` is a v2
  stub.
- **`providers.whisper.WhisperTranscriber` (Protocol)** — `async
  transcribe(audio: Path) → list[Cue]`. Cues use the same
  `pipeline.types.Cue` dataclass the VTT path produces, so downstream
  stages don't care which source produced them. `FasterWhisperTranscriber`
  is the only implementation in v1.1: lazy-imports `faster_whisper`,
  caches the model on the instance, runs sync inference via
  `asyncio.to_thread`. Defaults: `model="base.en"`, `device="cpu"`,
  `compute_type="int8"`. Since v1.3 the transcriber raises
  `NoSpeechDetectedError` (also exported from `providers.whisper`) when
  the materialized segment list is empty OR avg `no_speech_prob` > 0.6;
  the orchestrator catches it in Stage 2 and sets `status=error` with
  a friendly message rather than letting hallucinated cues poison
  downstream stages.

The factories `build_llm`, `build_vision`, `build_embedder`, and
`build_whisper` are the only sanctioned construction sites — the
orchestrator calls them and never instantiates clients directly.

## The 10-Stage Pipeline

`pipeline.pipeline.run_job(job_id, url, settings, jobs_root)` is the
single orchestrator. Stages (each preceded by a `_update(progress=...)`
manifest write):
1. Download video + VTT (`pipeline.download.download_video_and_captions`).
   No captions: if `whisper_fallback=False`, manifest error mentions
   `WHISPER_FALLBACK` and early return (NOT raise). If
   `whisper_fallback=True`, take the Whisper branch in Stage 2.
2. **Captions branch (one of two paths):**
   - **VTT path** (captions present): `parse_vtt` → `dedupe_rolling`.
   - **Whisper path** (no captions, fallback enabled): extract a 16 kHz
     mono WAV with `pipeline.audio.extract_audio` (ffmpeg `-vn -ac 1
     -ar 16000 -f wav` to `audio.wav`), then `await
     build_whisper(settings).transcribe(audio_path)` → `list[Cue]`.
     Both paths produce the same `list[Cue]` shape; downstream stages
     are identical.
3. Extract frames @ 1 fps with pHash dedup (`FixedFpsExtractor`,
   `hamming_max=6`).
4. Thumbnail frames to ~224px via `pipeline.frames.thumbnail_for_embedding`
   into `embed_thumbs/`, then embed those thumbs
   (`embedder.embed_images`); save `frame_embeddings.npy`. The
   thumbnail step is **mandatory** — original 720p frames blow past
   the Jina free tier's 100K-token/min budget. Originals stay in
   `frames/` for the result page.
5. LLM Pass 1 outline → `outline.json` (`llm_outline`, json_object +
   slice-fallback).
6. Embed step briefs (`embedder.embed_texts`).
7. Match frames to steps (`match`, pure; top-k=3, pad_sec=2,
   midpoint-nearest fallback for empty windows).
8. Caption only the union of winner frames (`caption_winners`,
   semaphore-bounded fanout, per-frame failure tolerated).
9. LLM Pass 2 refine each step (`llm_refine`, semaphore fanout, slice
   fallback, degraded path drops captions if refine fails; hard-coded
   `max_tokens=1500` to leave headroom for DeepSeek-style reasoning
   models).
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
  yet `done`. Each step on the result page carries a "Watch this step
  in the original video ↗" deep-link (built via `_video_deep_link`)
  that opens the source YouTube URL at the step's `start` time in a
  new tab (`target=_blank rel=noopener noreferrer`).
- `GET /job/{id}/frame/{name}.jpg` → serve `data/jobs/<id>/frames/{name}.jpg`.

Every route validates `job_id` via `_JOB_ID_RE`. Two accepted shapes:
`<11-char video id>_<6 hex>` (current, e.g. `dQw4w9WgXcQ_a1b2c3` — so
the URL itself names the source video) and `<12 hex>` (legacy v1.0/1.1
form, preserved for backward compat with jobs already on disk). The
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

Whisper-fallback settings:
- `WHISPER_FALLBACK` (`whisper_fallback: bool`, default `False`) —
  when `True`, the orchestrator transcribes audio locally instead of
  erroring on captionless videos. Strongly recommended for real
  YouTube traffic now that PO-tokens are required for most subtitle
  access (see Gotchas).
- `WHISPER_MODEL` (`whisper_model: str`, default `"base.en"`) — passed
  to `faster_whisper.WhisperModel`. Bigger models (`small.en`,
  `medium.en`) give better accuracy at higher CPU/RAM cost.

## Job Directory Layout

```
data/jobs/<job_id>/
  meta.json              ← Manifest (atomic write, single writer)
  video.mp4              ← yt-dlp output
  *.en.vtt               ← yt-dlp captions (absent on Whisper path)
  audio.wav              ← extract_audio output (only on Whisper path)
  frames/0000.jpg, …     ← FixedFpsExtractor (4-digit zero-pad)
  embed_thumbs/0000.jpg…  ← thumbnail_for_embedding output (~224px)
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
- Job IDs match `_JOB_ID_RE` in `server.py` — either `<11-char vid>_<6 hex>`
  (current) or `<12 hex>` (legacy). Frame filenames are 4-digit zero-pad.
- `pipeline._prompts.load_system_user` requires exactly one `## User`
  heading per prompt file; it raises `RuntimeError` otherwise.

## Gotchas
- **YouTube PO-token / Whisper is now the default**: mid-2024+ YouTube
  blocks subtitle download without a PO token, so the VTT path fails
  on most real videos. Operators should set `WHISPER_FALLBACK=1` in
  `.env`; otherwise `download_video_and_captions` will return
  `vtt=None` and the orchestrator will record a manifest error.
- **DeepSeek model names**: `deepseek-chat` is dead. Use
  `deepseek-v4-flash` (cheap, non-reasoning) or `deepseek-v4-pro`
  (reasoning, more expensive). Both ship CoT in `reasoning_content`
  which we ignore; v4-pro in particular needs the refine pass's
  `max_tokens=1500` headroom or the visible `content` returns empty.
  Pricing entries for all three are in `pricing.py`.
- **qwen-studio SSE shape**: each `data:` line is the literal next-token
  text, not a JSON object. `LLMClient` auto-detects this. Some strict
  providers 400 on `stream_options` — see `*_INCLUDE_USAGE` flags.
- **Jina free tier (100K tokens/min) is structurally tight**: 720p
  frames blow the budget — that's why the Stage 4 thumbnail step is
  mandatory. Even with 224px thumbs, longer videos can throttle;
  `JinaEmbedder` retries up to 5x on 429 honoring `Retry-After`,
  so embed-stage latency can spike during throttling. Production-scale
  operators should upgrade the tier or accept the wait.
- **Pricing zero**: missing model entries in `pricing.py` record zero
  and warn-once. Recorded cost will be a lower bound; check the log.
  Together's `meta-llama/Llama-Vision-Free` is intentionally $0/M
  (free tier) — that's not a missing entry, but the free tier has its
  own rate limits separate from billing.
- **Whisper model download on first run**: `FasterWhisperTranscriber`
  lazy-loads `faster_whisper.WhisperModel` on the first transcribe
  call, which downloads weights (~150MB for `base.en`) to the
  HuggingFace cache. The first Whisper-path job on a fresh host will
  pause for the download.
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
