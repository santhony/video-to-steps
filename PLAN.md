# video-to-steps — project plan

**Goal:** a web app that takes a YouTube URL for an instructional
video and returns a sequential list of written instructions, each
illustrated by the most relevant frame(s) from the video.

This document is the design plan written before any code, intended to
be implemented by a fresh Claude Code (or other coding agent) session
on a machine with bypass permissions, possibly different from the one
where the plan was drafted. The document is self-contained — every
decision needed to start coding is recorded here.

The app is designed to run in **three deployment modes** with the same
code: fully local (MLX + local LLM server), hybrid (local CLIP, cloud
LLM), and fully cloud-backed (vision-LLM captions + cloud LLM). The
choice is configuration, not a rewrite.

---

## 1. Decisions already made

| Decision | Choice | Rationale |
| --- | --- | --- |
| **Scope** | Standalone repo (not a tab in qwen-studio) | Cleaner separation; shareable and extractable. |
| **Frame-step matching** | Hybrid: timestamp window + visual-embedding cosine similarity | Timestamp alone fails when narrator-vs-visual lag is non-trivial; visual alone picks similar moments from unrelated parts. Hybrid restricts the visual search to ±N seconds of the step's spoken window. |
| **LLM passes** | Two: outline, then refine | Pass 1 segments the transcript into steps with time ranges; pass 2 turns each step's caption snippet into 1–3 polished instruction sentences. |
| **LLM provider abstraction** | OpenAI-compatible `/v1/chat/completions` everywhere | Most local servers (qwen-studio, Ollama via `/v1`, vLLM, llama.cpp server, LM Studio) and most cloud providers (OpenAI, DeepSeek, Groq, Together, Fireworks, OpenRouter) speak this format. Two settings: `LLM_BASE_URL`, `LLM_API_KEY` (optional). |
| **Visual-embedding provider abstraction** | Two interchangeable backends behind one `Embedder` interface: MLX CLIP (local) and vision-LLM captioning + text embeddings (cloud-friendly) | True cloud CLIP-style APIs are uneven; vision-LLM captioning ("describe what this frame shows") followed by a text embedding works through any chat-capable vision model and any text-embedding provider, which between them have broad coverage. |
| **Text-embedding provider abstraction** | OpenAI-compatible `/v1/embeddings` | Used in the cloud visual path and any future RAG-like additions. |
| **Caption source** | YouTube auto-captions via `yt-dlp`, with Whisper as an optional fallback | Most instructional videos have captions; Whisper is heavier. Plan implementation as optional flag, not v1 default. |
| **Storage** | Local disk under `data/jobs/{job_id}/` | Simple. S3 / object-store left as a v2 hook; the relevant boundary is documented in §10. |
| **Runtime model** | Single-process, in-process async tasks, JSON manifest on disk | One job at a time per worker; document the upper bound for v1. Multi-worker queue is v2. |
| **Server bind** | `127.0.0.1` by default, configurable to `0.0.0.0` via env for hosted deployment | Safe default; hosting requires explicit opt-in. |
| **Auth** | None in v1; deploy behind a reverse proxy or VPN when on `0.0.0.0` | Documented in §11 Hosting. |

## 2. Tech stack

Core (all modes):
- Python 3.11
- FastAPI + uvicorn
- Jinja2 + HTMX templates
- `yt-dlp` (video + caption download)
- `ffmpeg` (frame extraction; system binary)
- `httpx` (async HTTP for provider APIs)
- `numpy` (cosine similarity)
- `webvtt-py` (VTT caption parsing)
- `Pillow` (image read for vision-LLM payload)
- `pydantic` (config models)

Local-visual-embedding extras (when `EMBED_BACKEND=mlx_clip`):
- `mlx` ≥ 0.18
- `mlx-clip` (with `open_clip_torch` as a documented fallback if
  `mlx-clip` fails to install cleanly)

Cloud-visual-embedding path (when `EMBED_BACKEND=vision_caption`):
- No extra dependencies beyond the core; uses the same OpenAI-compatible
  `/v1/chat/completions` for captioning a frame (multimodal user
  message with an image part) and `/v1/embeddings` for embedding the
  captions.

System dependencies expected on the host:
- `ffmpeg` ≥ 4
- `python3.11`

## 3. Configuration

A single `.env` (loaded at startup) plus the same variables overridable
as process env vars. Pydantic model in `config.py`.

```
# ─── App ──────────────────────────────────────────────────────────────────────
APP_HOST=127.0.0.1
APP_PORT=8090
DATA_DIR=./data

# ─── LLM provider (OpenAI-compatible /v1/chat/completions) ────────────────────
# Default: local qwen-studio text server. To use cloud, set a provider URL
# and API key (e.g. https://api.openai.com, https://api.deepseek.com).
LLM_BASE_URL=http://127.0.0.1:8766
LLM_API_KEY=
LLM_MODEL=deepseek-v4-flash        # or gpt-4o-mini, deepseek-chat, etc.
LLM_PATH_CHAT=/chat                # qwen-studio uses /chat; cloud uses /v1/chat/completions
LLM_STREAM=true                    # qwen-studio streams; some cloud paths can disable
LLM_MAX_TOKENS=8192
LLM_TIMEOUT_SECONDS=300

# ─── Visual embedding backend ─────────────────────────────────────────────────
# One of: mlx_clip, openclip_torch, vision_caption
EMBED_BACKEND=mlx_clip

# Only used when EMBED_BACKEND=mlx_clip or openclip_torch:
CLIP_MODEL=openai/clip-vit-base-patch32

# Only used when EMBED_BACKEND=vision_caption — the cloud-friendly mode:
VISION_BASE_URL=                   # blank means reuse LLM_BASE_URL
VISION_API_KEY=                    # blank means reuse LLM_API_KEY
VISION_MODEL=gpt-4o-mini           # any chat model accepting image_url parts
TEXT_EMBED_BASE_URL=               # blank means reuse LLM_BASE_URL
TEXT_EMBED_API_KEY=                # blank means reuse LLM_API_KEY
TEXT_EMBED_MODEL=text-embedding-3-small
TEXT_EMBED_PATH=/v1/embeddings
```

The three intended modes select themselves by which variables are set:

- **Mode A — fully local.** `LLM_BASE_URL=http://127.0.0.1:8766`,
  `LLM_API_KEY=` empty, `EMBED_BACKEND=mlx_clip`. No internet.
- **Mode B — hybrid (local CLIP, cloud LLM).** `LLM_BASE_URL` points
  at a cloud provider, `LLM_PATH_CHAT=/v1/chat/completions`,
  `LLM_API_KEY` set, `EMBED_BACKEND=mlx_clip`. Useful on Apple
  Silicon machines that have a CLIP-capable Metal but want stronger
  LLM quality.
- **Mode C — cloud-only.** All providers cloud,
  `EMBED_BACKEND=vision_caption`. Runs on a vanilla Linux server with
  no GPU. Each frame is captioned by a vision LLM, captions are
  text-embedded, step text is text-embedded, cosine similarity as
  before.

## 4. Tech stack — provider abstractions

### `providers/llm.py`

```python
class LLMClient:
    def __init__(self, base_url: str, api_key: str | None, model: str,
                 path: str = "/v1/chat/completions",
                 stream: bool = True): ...
    async def chat(self, messages: list[dict], *,
                   max_tokens: int = 4096,
                   response_format: dict | None = None) -> str: ...
```

Implementation:
- POST to `base_url + path`.
- If `api_key` is set, send `Authorization: Bearer {api_key}`.
- If `stream=true`, accumulate SSE `data:` lines into a string. Recognize
  two stream shapes: qwen-studio's `data: <token>\n\n` (just text), and
  OpenAI's `data: {"choices":[{"delta":{"content":"..."}}]}\n\n`.
  Auto-detect by trying to JSON-parse each `data:` line; if it parses
  and has `choices[0].delta.content`, use that; otherwise treat the
  line as raw text.
- If `stream=false`, parse single JSON response, return `choices[0].message.content`.
- Always strip `<think>...</think>` blocks from the returned string
  before handing to the caller (DS4 emits them).

### `providers/embed.py`

```python
class Embedder(Protocol):
    def name(self) -> str: ...
    async def embed_images(self, paths: list[Path]) -> np.ndarray:
        """Return float32 (n, d) L2-normalized embeddings."""
    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return float32 (n, d) L2-normalized embeddings."""
```

Two concrete implementations:

**`providers/embed_mlx_clip.py`** — uses `mlx_clip` to embed both images
and texts in a shared CLIP space.

**`providers/embed_vision_caption.py`** — pipeline:
1. For each frame: call vision LLM with a system prompt "Describe what
   is shown in this still frame in one detailed sentence focused on
   actions, tools, and materials. Do not speculate beyond the image."
   plus the user message: `[{"type": "image_url", "image_url": {"url":
   "data:image/jpeg;base64,<bytes>"}}]`. Cache the caption per frame.
2. Embed each caption via `/v1/embeddings`.
3. For step texts, embed directly via `/v1/embeddings`.
4. Both vectors live in the same text-embedding space; cosine similarity
   works.

A factory `build_embedder(config) -> Embedder` reads `EMBED_BACKEND` and
returns the right instance. Pipeline code never branches on backend.

### Why captioning instead of a hosted CLIP-as-a-service

There is no broadly available, well-priced, OpenAI-compatible "embed
this image" cloud API at the time of writing. Multimodal embedding APIs
(Voyage, Cohere, Jina) exist but each requires its own SDK and a
separate paid account. Reusing the LLM chat endpoint with a vision
model for captioning gives us a working cloud path with no extra
account, and the framing space (text-embedding) is also commodity. The
quality is comparable to CLIP for instructional-video frame matching
because the step text and frame caption are both natural-language
descriptions of the same scene.

## 5. Repository layout

```
video-to-steps/
  README.md
  PLAN.md
  .env.example
  requirements.txt
  setup.sh                     # creates venv, installs core + selected backend
  start.sh                     # launches server
  stop.sh
  Dockerfile                   # for Mode C / hosted deployment
  .gitignore
  server.py                    # FastAPI app, routes, lifespan
  config.py                    # pydantic Settings
  providers/
    __init__.py
    llm.py
    embed.py                   # Embedder protocol + factory
    embed_mlx_clip.py
    embed_openclip_torch.py    # fallback if mlx_clip unavailable
    embed_vision_caption.py
  pipeline/
    __init__.py
    download.py                # yt-dlp wrapper
    captions.py                # VTT -> [Cue]
    frames.py                  # ffmpeg frame extraction
    match.py                   # hybrid timestamp+embedding matching
    pipeline.py                # orchestrator
    types.py                   # Cue, Step, Frame, Match, Manifest
  prompts/
    outline.md                 # pass-1 system + user template
    refine.md                  # pass-2 system + user template
    vision_caption.md          # frame-captioning prompt (Mode C)
  templates/
    base.html
    index.html
    job.html
    result.html
  static/
    css/main.css
    js/htmx.min.js
  data/                        # gitignored
    jobs/{job_id}/
      meta.json                # url, status, started_at, finished_at, error, progress
      video.mp4
      captions.vtt
      frames/####.jpg
      frame_embeddings.npy     # cached
      frame_captions.json      # only in vision_caption mode
      outline.json
      steps.json
```

## 6. Data shapes (pipeline/types.py)

```python
@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float

@dataclass(frozen=True)
class StepOutline:
    index: int
    start: float
    end: float
    brief: str

@dataclass(frozen=True)
class Step:
    index: int
    start: float
    end: float
    instruction: str
    frames: list[Frame]

@dataclass
class Manifest:
    job_id: str
    url: str
    mode: str                  # "A" | "B" | "C" — recorded for debugging
    status: Literal["pending","downloading","extracting","embedding",
                    "outlining","refining","matching","done","error"]
    error: str | None
    started_at: str
    finished_at: str | None
    progress: float
    progress_note: str
```

## 7. Pipeline stages — contracts

Each stage is testable in isolation; the orchestrator chains them.

### 7.1 download.py

`download_video_and_captions(url, job_dir) -> (video_path, vtt_path | None)`

yt-dlp options: `format="best[ext=mp4][height<=720]/best[height<=720]/best"`,
`writesubtitles=True, writeautomaticsub=True, subtitleslangs=["en"],
subtitlesformat="vtt"`, quiet.

If no captions and `WHISPER_FALLBACK=true`, call `whisper.cpp` or
`faster-whisper` (decide at implementation time; document choice).
Otherwise raise — caller surfaces error.

### 7.2 captions.py

`parse_vtt(path) -> list[Cue]` and `dedupe_rolling(cues) -> list[Cue]`.
YouTube auto-captions repeat lines with growing timestamps; collapse.

### 7.3 frames.py

`extract_frames(video, frames_dir, fps=1.0, max_height=720) -> list[Frame]`

```
ffmpeg -nostdin -loglevel error -y -i {video} \
  -vf "fps={fps},scale=-2:{max_height}" -qscale:v 4 {dir}/%04d.jpg
```

Filename indices are 1-based on disk; in-memory `Frame.index` is 0-based.
Confine the +1 to filename formatting.

### 7.4 LLM passes (`pipeline/pipeline.py` orchestrates; prompts in `prompts/`)

**Pass 1 — outline.** Inputs: deduped, timestamped transcript as a
single block, one line per cue: `[MM:SS] text`. Output: a JSON array
of `{index, start, end, brief}`. Use `response_format={"type":
"json_object"}` if the provider supports it (OpenAI, DeepSeek do).
Otherwise robustly slice from first `[` to matching final `]`.

**Pass 2 — refine.** Per step, fan out concurrent calls (cap at 4
in-flight) with the step's brief + the cue text contained in the step's
time range. Output: 1–3 polished sentences, second-person imperative.

### 7.5 Visual matching

```python
async def embed_step_visuals(steps, embedder, job_dir): ...
async def match(steps, frame_embeddings, frame_index_to_ts,
                step_embeddings, *, pad_sec=5.0, top_k=2): ...
```

For each step:
- Candidate frames: `step.start - pad <= ts <= step.end + pad`.
- Score: `frame_emb @ step_emb`. Pick top-k.
- Fallback if empty candidate set: nearest frame to step midpoint.

In **Mode C**, "frame embeddings" are actually text embeddings of the
frame captions. Score is still cosine; the math is identical.

## 8. Orchestrator (`pipeline/pipeline.py`)

```python
async def run_job(job_id, url, settings, jobs_root) -> None:
    job_dir = jobs_root / job_id
    manifest = ManifestFile(job_dir / "meta.json")
    try:
        manifest.update(status="downloading", progress=0.05)
        video, vtt = download_video_and_captions(url, job_dir)
        if not vtt and not settings.whisper_fallback:
            raise RuntimeError("no captions and whisper fallback disabled")
        cues = dedupe_rolling(parse_vtt(vtt))

        manifest.update(status="extracting", progress=0.15)
        frames = extract_frames(video, job_dir / "frames",
                                fps=settings.frames_fps)

        manifest.update(status="embedding", progress=0.30,
                        progress_note=f"embedder={embedder.name()}")
        embedder = build_embedder(settings)
        frame_paths = [job_dir / "frames" / f"{f.index+1:04d}.jpg" for f in frames]
        frame_emb = await embedder.embed_images(frame_paths)
        np.save(job_dir / "frame_embeddings.npy", frame_emb)

        manifest.update(status="outlining", progress=0.55)
        outline = await llm_outline(cues, settings)
        (job_dir / "outline.json").write_text(json.dumps([asdict(s) for s in outline]))

        manifest.update(status="refining", progress=0.75)
        refined = await llm_refine(outline, cues, settings, max_in_flight=4)

        manifest.update(status="matching", progress=0.90)
        text_emb = await embedder.embed_texts([s.instruction for s in refined])
        picks = match(outline, frame_emb,
                      {f.index: f.timestamp for f in frames},
                      text_emb, pad_sec=settings.match_pad_sec,
                      top_k=settings.frames_per_step)
        steps = [...]
        (job_dir / "steps.json").write_text(...)
        manifest.update(status="done", progress=1.0, finished_at=now_iso())
    except Exception as e:
        manifest.update(status="error", error=str(e), finished_at=now_iso())
        raise
```

## 9. Server routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Render `index.html` |
| POST | `/process` | Validate URL, allocate `job_id`, spawn task, redirect to `/job/{id}` |
| GET | `/job/{id}` | Render `job.html` (status page) |
| GET | `/job/{id}/status` | Partial HTML fragment with progress, polled every 2s via HTMX |
| GET | `/job/{id}/result` | Render `result.html` from `steps.json` |
| GET | `/job/{id}/frame/{n}.jpg` | Serve `frames/####.jpg` |

Bind to `settings.app_host:settings.app_port` (`127.0.0.1:8090` by
default; set `APP_HOST=0.0.0.0` for hosted).

## 10. Storage abstraction (small but important for v2)

For v1, all reads/writes go through helper functions in
`pipeline/storage.py`:

```python
def job_dir(job_id: str) -> Path: ...
def write_bytes(path: Path, data: bytes) -> None: ...
def read_bytes(path: Path) -> bytes: ...
def write_json_atomic(path: Path, obj) -> None: ...
def open_frame(path: Path) -> bytes: ...
```

In v1 these wrap local fs. v2 can swap to an S3-backed implementation
without touching the pipeline modules.

## 11. Hosting

Two paths documented in README:

**A. Stay local.** `./setup.sh && ./start.sh`. Browser at
`http://127.0.0.1:8090`.

**B. Hosted deployment** (any VPS that can run Python and ffmpeg, no
GPU required if using Mode C):
1. `git clone … && cd video-to-steps`
2. `./setup.sh`
3. `.env`: set `APP_HOST=0.0.0.0`, set `LLM_*` and embedding vars to
   cloud values, set `EMBED_BACKEND=vision_caption`.
4. Put it behind a reverse proxy (nginx/Caddy) that terminates TLS and
   adds whatever auth you want (basic auth is fine for personal use).
   The app itself has no auth.
5. `./start.sh` or run under systemd with the unit file given in the
   README.

A `Dockerfile` is provided:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV APP_HOST=0.0.0.0 APP_PORT=8090
EXPOSE 8090
CMD ["python", "server.py"]
```

(Dockerfile installs only core deps. For Mode A/B with `mlx_clip`, run
on a Mac instead — `mlx` is Apple Silicon only.)

## 12. Setup script

```sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv venv
./venv/bin/pip install -U pip wheel
./venv/bin/pip install -r requirements.txt
# Backend-specific extras
case "${EMBED_BACKEND:-mlx_clip}" in
    mlx_clip)
        ./venv/bin/pip install "mlx>=0.18" "mlx-clip" || \
            echo "[!] mlx-clip install failed. Set EMBED_BACKEND=openclip_torch or vision_caption in .env, then re-run."
        ;;
    openclip_torch)
        ./venv/bin/pip install torch open_clip_torch
        ;;
    vision_caption)
        echo "Cloud captioning mode — no extra deps."
        ;;
    *)
        echo "Unknown EMBED_BACKEND: ${EMBED_BACKEND}"; exit 1
        ;;
esac
cp -n .env.example .env || true
echo "Setup complete. Edit .env for provider settings, then ./start.sh"
```

`requirements.txt` core:

```
fastapi>=0.110
uvicorn[standard]>=0.27
jinja2>=3.1
httpx>=0.27
yt-dlp>=2025.1.0
webvtt-py>=0.5.1
numpy>=1.26
Pillow>=10
pydantic>=2
python-dotenv>=1
```

## 13. Implementation order

1. Scaffold (directories, `.gitignore`, empty files, README skeleton).
2. `config.py` + `.env.example`.
3. `providers/llm.py` with the dual-stream auto-detect. Smoke test
   against the configured `LLM_BASE_URL`.
4. `providers/embed.py` Protocol + factory.
5. Implement **one** embedder first — pick `mlx_clip` if on Apple
   Silicon, else `vision_caption`. Get end-to-end working with that
   one before adding the second.
6. `pipeline/types.py`.
7. `pipeline/download.py` and `pipeline/captions.py`. Test on a short
   instructional video.
8. `pipeline/frames.py`.
9. `pipeline/match.py` (pure function, easy to unit-test).
10. `pipeline/pipeline.py` orchestrator (writes manifest progress).
11. `server.py` + templates.
12. Implement the second embedder so both modes work.
13. `Dockerfile`, README hosting section, smoke-test in cloud mode if
    feasible.
14. Push to `santhony/video-to-steps`.

## 14. Out of scope for v1

- Whisper fallback (mention in code path, don't implement)
- Multi-job queue, persistent task recovery across restarts
- S3 / object-store for `data/jobs/`
- Authentication (defer to reverse proxy)
- Diff/refine UI for the generated instructions
- Export to PDF / Markdown
- Non-YouTube sources (yt-dlp covers many but the UI only mentions
  YouTube)

## 15. Known risks / footguns

- **`mlx-clip` install reliability.** Spotty release history. The
  setup script tolerates failure and instructs the user to fall back
  to `openclip_torch` or `vision_caption`. Document this prominently
  in README.
- **YouTube captions absent.** `yt-dlp` returns no `.vtt`. v1 errors
  out with a clear message; v2 plugs in Whisper.
- **DS4 `<think>` leakage.** Always strip `<think>...</think>` from
  LLM output before JSON parsing or persistence.
- **`response_format=json_object`.** Supported by OpenAI and DeepSeek
  cloud, but not by all OpenAI-compatible servers. Always pair it
  with a manual JSON-slice fallback ("locate first `[`, last matching
  `]`, parse").
- **Vision-LLM rate limits in Mode C.** A 30-minute video at 1 fps is
  1800 captioning calls. Use `asyncio.gather` with a sane semaphore
  (~16 concurrent) and retries with exponential backoff. Document
  expected cost per video at OpenAI/DeepSeek vision rates.
- **Off-by-one on frame numbering.** ffmpeg writes `0001.jpg…`;
  in-memory `Frame.index` is 0-based. Confine the +1 to disk-side.
- **Bind safety.** Default `127.0.0.1`. Document that switching to
  `0.0.0.0` must be paired with a reverse proxy that handles auth.
- **Frame URL leakage if hosted.** Job IDs are unguessable UUIDs;
  without auth, anyone with the URL can view the result. Acceptable
  for personal hosted use behind basic auth; document this constraint.

## 16. Testing expectations

Do not claim done until the end-to-end smoke test works on **one**
short real instructional video in the chosen primary mode. Suggested
candidates: a 3–5 minute knot-tying tutorial, a short recipe
walkthrough, a short woodworking step demo. Sign of success: the
result page shows N ordered steps, each with at least one image, and
the images visibly correspond to the step text. If running in Mode C,
verify that the vision-captioning path produces sensible captions for
a handful of frames before running on a full video.

After v1 smoke test passes, run the same video through the *other*
configured mode to verify both paths produce comparable output.

---

End of plan.
