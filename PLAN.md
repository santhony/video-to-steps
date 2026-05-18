# video-to-steps — project plan

**Goal:** a local web app that takes a YouTube URL for an instructional
video and returns a sequential list of written instructions, each
illustrated by the most relevant frame(s) from the video.

This document is the design plan written before any code, intended to be
implemented by a fresh Claude Code (or other coding agent) session on a
machine with bypass permissions, possibly different from the one where
the plan was drafted. The document is self-contained — every decision
needed to start coding is recorded here.

---

## 1. Decisions already made

| Decision | Choice | Rationale |
| --- | --- | --- |
| **Scope** | Standalone repo (not a tab in qwen-studio) | Cleaner separation; the user wants this shareable and extractable. |
| **Frame-step matching** | Hybrid: timestamp window + CLIP cosine similarity | Timestamp alone fails when narrator-vs-visual lag is non-trivial; CLIP alone picks visually similar moments from unrelated parts of the video. Hybrid restricts the CLIP search to ±N seconds of the step's spoken window. |
| **LLM passes** | Two: outline, then refine | Pass 1 segments the transcript into steps with time ranges; pass 2 turns each step's raw caption snippet into 1–3 polished instruction sentences. Costs more tokens but gives better wording. |
| **LLM endpoint** | Pluggable; defaults to qwen-studio text server (`http://127.0.0.1:8766/chat`) | Local-only by default. Works regardless of which backend qwen-studio is using (DS4 / Ollama / MLX). User can point it elsewhere with an env var. |
| **CLIP backend** | MLX (Apple Silicon) | Mirrors the qwen-studio approach (`mlx_lm`, `mlx_embeddings`). Avoids dragging in PyTorch + MPS. |
| **Caption source** | YouTube auto-captions via `yt-dlp`, with no Whisper fallback in v1 | Most instructional videos have captions; Whisper adds ~3 GB of model weight and minutes of runtime. Mark as a v2 add-on. |
| **Runtime model** | Single-machine, no queue infra | One job at a time, persisted as a directory under `data/jobs/{job_id}/`. Simple JSON status file polled by the browser. |

## 2. Tech stack

- Python 3.11 (matches qwen-studio venvs)
- FastAPI + uvicorn (HTTP)
- Jinja2 + HTMX (templates and polling, same pattern as qwen-studio)
- `yt-dlp` (video + caption download)
- `ffmpeg` (frame extraction; system binary, already installed)
- `mlx_clip` from PyPI (CLIP image+text embeddings on Metal)
  - Fallback: if `mlx_clip` has issues at install, use `open_clip_torch`
    with `torch` MPS backend. Decide at setup time; document chosen path.
- `numpy` for cosine similarity over frame embeddings
- `httpx` for LLM HTTP calls
- No database; per-job JSON manifests on disk

System dependencies expected on the host:
- `ffmpeg` ≥ 4 (we have 7.1.1 on the dev box)
- `python3.11`
- An LLM HTTP endpoint reachable from the app (default localhost:8766)

## 3. Repository layout

```
video-to-steps/
  README.md
  PLAN.md                      # this document, after rename
  requirements.txt
  setup.sh                     # creates venv, installs deps
  start.sh                     # launches server on 127.0.0.1:8090
  stop.sh
  .gitignore                   # ignores venv/, data/jobs/, __pycache__, .DS_Store
  server.py                    # FastAPI app, routes, lifespan
  pipeline/
    __init__.py
    download.py                # yt-dlp wrapper
    captions.py                # VTT -> [{start,end,text}, ...]
    frames.py                  # ffmpeg frame extraction
    clip_embed.py              # MLX CLIP wrapper
    llm.py                     # HTTP client for OpenAI-compatible /chat
    match.py                   # hybrid timestamp+CLIP matching
    pipeline.py                # orchestrator that runs all stages with progress
    types.py                   # dataclasses: Cue, Step, Frame, Match, Manifest
  templates/
    base.html                  # nav, status, htmx + main.css
    index.html                 # input form
    job.html                   # status page (htmx-polled)
    result.html                # final rendered instructions
  static/
    css/main.css
    js/htmx.min.js             # copy from qwen-studio
  data/
    jobs/                      # one dir per job; gitignored
      {job_id}/
        meta.json              # url, status, started_at, finished_at, error
        video.mp4
        captions.vtt
        frames/####.jpg
        frame_embeddings.npy   # (n_frames, embed_dim) float32
        outline.json           # pass-1 LLM result
        steps.json             # final list with matched frames
```

## 4. Data shapes (pipeline/types.py)

```python
@dataclass(frozen=True)
class Cue:
    start: float          # seconds
    end: float
    text: str             # one line of caption

@dataclass(frozen=True)
class Frame:
    index: int            # 0-based, matches filename ####.jpg
    timestamp: float      # seconds into the video

@dataclass(frozen=True)
class StepOutline:
    index: int            # 0-based
    start: float
    end: float
    brief: str            # one-line description from pass-1 LLM

@dataclass(frozen=True)
class Step:
    index: int
    start: float
    end: float
    instruction: str      # refined text from pass-2 LLM (1-3 sentences)
    frames: list[Frame]   # 1-2 frames from pass-3 matching

@dataclass
class Manifest:
    job_id: str
    url: str
    status: Literal["pending","downloading","extracting","embedding",
                    "outlining","refining","matching","done","error"]
    error: str | None
    started_at: str       # ISO8601
    finished_at: str | None
    progress: float       # 0.0 to 1.0
    progress_note: str    # human-readable current activity
```

## 5. Pipeline stages — detailed contracts

Each stage is a pure(-ish) function that reads inputs from `job_dir`,
writes outputs to `job_dir`, and updates the `Manifest` in place via a
helper. Stages do **not** call each other directly; the orchestrator
chains them. This keeps each stage testable in isolation and allows
re-running a single stage by deleting its outputs.

### 5.1 download.py

```python
def download_video_and_captions(url: str, job_dir: Path) -> tuple[Path, Path | None]:
    """Returns (video_path, captions_path or None if no captions available)."""
```

Implementation:
- `yt_dlp.YoutubeDL({...})` with options:
  - `format`: `"best[ext=mp4][height<=720]/best[height<=720]/best"`
  - `outtmpl`: `str(job_dir / "video.%(ext)s")`
  - `writesubtitles`: True, `writeautomaticsub`: True
  - `subtitleslangs`: `["en"]`, `subtitlesformat`: `"vtt"`
  - `quiet`: True, `no_warnings`: True
- Locate the downloaded `.vtt`: glob `job_dir/*.en.vtt` then `*.vtt`. If
  none, return `(video_path, None)` — caller decides to error out.

Error modes:
- Invalid URL → raise `ValueError("not a recognizable YouTube URL")`
- Age-restricted/private → yt_dlp raises; caller catches and writes to
  `Manifest.error`.

### 5.2 captions.py

```python
def parse_vtt(vtt_path: Path) -> list[Cue]:
    """Parse a WEBVTT file into a list of cues sorted by start time."""

def dedupe_rolling(cues: list[Cue]) -> list[Cue]:
    """
    YouTube auto-captions emit a rolling window where the same line
    reappears across consecutive cues with growing timestamps. Collapse:
    if cue N+1's text is a prefix-or-equal of cue N's text, drop N+1.
    Returns the cleaned list.
    """
```

Use the `webvtt-py` package (pure-Python, no external deps). Add to
requirements.

### 5.3 frames.py

```python
def extract_frames(video_path: Path, frames_dir: Path, fps: float = 1.0,
                   max_height: int = 720) -> list[Frame]:
    """
    Run ffmpeg to extract `fps` frames per second, scaled to max_height
    pixels. Save as zero-padded ####.jpg in frames_dir. Return a list
    of Frame(index, timestamp) where timestamp = index / fps.
    """
```

ffmpeg invocation:

```
ffmpeg -nostdin -loglevel error -y -i {video} \
  -vf "fps={fps},scale=-2:{max_height}" \
  -qscale:v 4 \
  {frames_dir}/%04d.jpg
```

(ffmpeg starts numbering at 1; subtract 1 when computing 0-based index,
or just rename. Pick one convention and stick to it — `index = N - 1`
is simplest.)

### 5.4 clip_embed.py

```python
class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"): ...
    def embed_images(self, paths: list[Path]) -> np.ndarray:  # (n, d) float32
    def embed_texts(self, texts: list[str]) -> np.ndarray:    # (n, d) float32
```

Implementation notes:
- Use `mlx_clip` if available. Its API is roughly:
  ```python
  from mlx_clip import load, encode_image, encode_text
  model, processor = load("openai/clip-vit-base-patch32")
  vec = encode_image(model, processor, image_path)
  ```
  (Confirm exact API at install time; this package's API has been mildly
  unstable in past releases.)
- Embeddings L2-normalized so cosine similarity is a dot product.
- Batch image embedding: 32 at a time to keep memory predictable.
- Cache: after first call, persist `frame_embeddings.npy` in job dir;
  next run reuses if frame count matches.

### 5.5 llm.py

```python
def chat(messages: list[dict], *, max_tokens: int = 4096,
         endpoint: str = "http://127.0.0.1:8766") -> str:
    """Send a chat request, accumulate the streamed response, return text."""
```

Wire format: same SSE contract qwen-studio uses (`data: <token>\n\n`,
ending with `data: [DONE]\n\n`). Re-assemble into one string. Strip
`<think>...</think>` blocks from the result (DS4 returns reasoning by
default; we don't need it in the final step text).

Endpoint configured via env: `LLM_ENDPOINT` defaults to
`http://127.0.0.1:8766`. Tools are not used.

### 5.6 Pass 1 prompt — outline

System message:

> You are converting an instructional video transcript into a structured
> outline. The user message contains the transcript with timestamps in
> `[MM:SS]` format. Produce a JSON array of steps. Each step has:
> `index` (0-based int), `start` (seconds), `end` (seconds), and `brief`
> (one short sentence describing what is done in this step). Group
> consecutive caption cues that describe one coherent action.
> Aim for 5–25 steps for a typical 10-minute video. Output ONLY the
> JSON array — no prose, no markdown fences.

User message: timestamped transcript, each line formatted
`[MM:SS] text` derived from the deduped cues.

Robustness: the model may emit extra prose. Locate the first `[` and
the matching final `]`, slice, parse. If parsing fails, retry once with
a stricter reminder. If it fails twice, surface the failure to the user.

### 5.7 Pass 2 prompt — refine

For each outline step, call the LLM with:

System: "You are rewriting a single step from a video transcript into
clear, actionable instructions for a written guide. Output 1–3
sentences in second-person imperative voice. No markdown headers, no
numbering — just the instruction text. Do not invent details not
present in the snippet."

User:

```
Step brief: {step.brief}
Spoken content during this step:
{joined caption text within step.start..step.end}
```

This pass is parallel-safe (each step is independent). Run with
`asyncio.gather` over an `httpx.AsyncClient`, capped at 4 in-flight to
not melt the single-graph DS4 worker.

### 5.8 match.py

```python
def match_frames_to_steps(
    steps: list[StepOutline],
    frame_embeddings: np.ndarray,
    frame_index_to_ts: dict[int, float],
    step_texts: list[str],
    text_embeddings: np.ndarray,
    *,
    pad_sec: float = 5.0,
    top_k: int = 2,
) -> list[list[int]]:
    """Return, for each step, the top-k frame indices."""
```

For step `i`:
1. Candidate set: frame indices with `step.start - pad <= ts <= step.end + pad`.
2. Score each candidate as `frame_emb @ text_emb[i]` (already
   L2-normalized so this is cosine).
3. Take the top `top_k`.

If the candidate set is empty (very short step), fall back to the
single frame closest to `(step.start + step.end) / 2`.

### 5.9 Orchestrator

`pipeline/pipeline.py`:

```python
async def run_job(job_id: str, url: str, jobs_root: Path) -> None:
    job_dir = jobs_root / job_id
    manifest = ManifestFile(job_dir / "meta.json")
    try:
        manifest.update(status="downloading", progress=0.05)
        video, vtt = download_video_and_captions(url, job_dir)
        if not vtt:
            raise RuntimeError("no captions available for this video")
        cues = dedupe_rolling(parse_vtt(vtt))

        manifest.update(status="extracting", progress=0.15)
        frames = extract_frames(video, job_dir / "frames")

        manifest.update(status="embedding", progress=0.30)
        embedder = CLIPEmbedder()
        frame_paths = [job_dir / "frames" / f"{f.index+1:04d}.jpg" for f in frames]
        frame_emb = embedder.embed_images(frame_paths)
        np.save(job_dir / "frame_embeddings.npy", frame_emb)

        manifest.update(status="outlining", progress=0.55)
        outline = await llm_outline(cues)
        (job_dir / "outline.json").write_text(json.dumps([asdict(s) for s in outline]))

        manifest.update(status="refining", progress=0.75)
        refined = await llm_refine(outline, cues, max_in_flight=4)

        manifest.update(status="matching", progress=0.90)
        text_emb = embedder.embed_texts([s.instruction for s in refined])
        picks = match_frames_to_steps(outline, frame_emb,
                                       {f.index: f.timestamp for f in frames},
                                       [s.instruction for s in refined], text_emb)
        steps = [
            Step(index=o.index, start=o.start, end=o.end,
                 instruction=r.instruction,
                 frames=[frames[i] for i in picks[k]])
            for k, (o, r) in enumerate(zip(outline, refined))
        ]
        (job_dir / "steps.json").write_text(json.dumps([asdict(s) for s in steps], default=str))
        manifest.update(status="done", progress=1.0,
                        finished_at=now_iso())
    except Exception as e:
        manifest.update(status="error", error=str(e),
                        finished_at=now_iso())
        raise
```

Each `manifest.update` writes atomically (write to `.tmp`, rename).

## 6. Server — server.py

Routes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Render `index.html` (URL input form) |
| POST | `/process` | Validate URL, allocate `job_id` (uuid4 short), kick off `asyncio.create_task(run_job(...))`, redirect to `/job/{id}` |
| GET | `/job/{id}` | Render `job.html`, which polls `/job/{id}/status` every 2s via HTMX |
| GET | `/job/{id}/status` | Return a partial HTML fragment showing status + progress bar + note; if `done`, the fragment includes a button/link to `/job/{id}/result` |
| GET | `/job/{id}/result` | Render `result.html` from `steps.json`. Each step: instruction text + 1-2 `<img>` referencing `/job/{id}/frame/{n}.jpg` |
| GET | `/job/{id}/frame/{n}.jpg` | Serve `frames/####.jpg` from disk |

Port: `127.0.0.1:8090` (next free local port; doesn't collide with
qwen-studio's 8080/8765/8766/8767).

Lifespan: maintain a dict `app.state.tasks: dict[job_id, asyncio.Task]`
so a server restart doesn't lose in-flight jobs silently. (For v1, jobs
killed by restart simply show their stale "downloading" status until
the user re-submits. Document this.)

## 7. UI

`index.html`: a clean form with one input — the YouTube URL — and a
"Process" button.

`job.html`: a single card showing:
- The URL being processed
- A progress bar (server fragment driven by `hx-get="/job/{id}/status"
  hx-trigger="load, every 2s" hx-swap="outerHTML"`)
- When status is "done", swap in a link to the result page

`result.html`: ordered list. Each `<li>` has:
- Step number and 1–3 sentence instruction (left column on desktop, top
  on mobile)
- One or two `<img>` tags, lazy-loaded, click to enlarge to actual size

Aesthetic: borrow the qwen-studio CSS variables (dark theme, accent
purple) by copying the relevant `:root` block. Keep `main.css` short.

## 8. Setup script (`setup.sh`)

```sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv venv
./venv/bin/pip install -U pip wheel
./venv/bin/pip install -r requirements.txt
echo "Setup complete. Run ./start.sh to launch."
```

`requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.27
jinja2>=3.1
httpx>=0.27
yt-dlp>=2025.1.0
webvtt-py>=0.5.1
numpy>=1.26
mlx>=0.18
mlx-clip>=0.1
Pillow>=10
```

If `mlx-clip` doesn't install or load cleanly, swap to:

```
torch
open_clip_torch
```

and rewrite `clip_embed.py` to use the PyTorch CLIP path on MPS.

## 9. start.sh / stop.sh

Same pattern as qwen-studio: `nohup` the server, write pid to
`data/server.pid`, log to `data/server.log`. Stop reads pid, SIGTERM,
waits, SIGKILL fallback.

## 10. Implementation order

For an agent implementing this end-to-end:

1. Scaffold the repo (directories, `.gitignore`, empty files).
2. `requirements.txt` + `setup.sh`; run setup; confirm `mlx-clip`
   actually installs. If it fails, switch to PyTorch path and document.
3. `pipeline/types.py`.
4. `pipeline/download.py` and `pipeline/captions.py`. Smoke-test on a
   single short instructional video (e.g. a 3-minute knot-tying clip).
5. `pipeline/frames.py`. Smoke-test produces N jpgs.
6. `pipeline/clip_embed.py`. Verify by embedding 10 frames and
   measuring shape + L2 norm.
7. `pipeline/llm.py` (basic chat wrapper). Verify against localhost:8766.
8. Outline + refine prompts. Verify on the test video — print the JSON
   outline and inspect by eye.
9. `pipeline/match.py`. Print step → picked frames mapping.
10. Orchestrator (`pipeline/pipeline.py`) wiring all of the above
    together with manifest progress writes.
11. `server.py` + templates. Manual browser test.
12. README documenting setup + run + the one external dependency
    (a local LLM at port 8766, or wherever).
13. Create the GitHub repo (`gh repo create santhony/video-to-steps
    --private --source . --remote origin --push`), initial commit.

## 11. Out of scope for v1 (note for v2)

- Whisper transcription for videos without captions
- Authentication / multi-user
- Persistent job queue across server restarts
- Diff/refine UI for editing the generated instructions
- Export to PDF / markdown
- Non-YouTube sources (the download stage is yt-dlp's broad coverage
  but is currently called with only YouTube URLs in mind; widening is
  trivial)

## 12. Known risks / footguns

- **mlx-clip install reliability.** Has had spotty releases on PyPI.
  If `pip install mlx-clip` fails, fall back to `open_clip_torch` —
  add the alternative path to the `clip_embed.py` from the start.
- **YouTube without auto-captions.** Most popular instructional content
  has them. If absent, we error out cleanly — no Whisper in v1.
- **DS4 timing.** For a 30-minute transcript, pass 1 may produce a
  large JSON; ensure `max_tokens` is generous (4–8k). The refine pass
  is small per call.
- **`<think>` leakage.** If using DS4, the model emits a thinking
  block before its actual output. The LLM client must strip it before
  JSON parsing in pass 1 and before persistence in pass 2.
- **Frame index off-by-one.** ffmpeg starts at 1; the data model uses
  0-based. Confine the +1 to disk filenames (`f"{i+1:04d}.jpg"`); in
  memory, always 0-based.
- **Long videos.** A 60-minute video at 1 fps is 3600 frames. CLIP
  batches of 32 → ~110 batches → minutes of MLX compute. Acceptable
  for v1. Document it.
- **Frame URL leakage.** Job IDs are unguessable UUIDs; no auth otherwise.
  Bind to 127.0.0.1 only.

## 13. Testing expectations

For the agent implementing: don't claim done until the end-to-end
smoke test works on **one** short real instructional video. Example
candidates (any one of these):

- A 3–5 minute knot-tying tutorial
- A short recipe walkthrough
- A short woodworking step demo

Sign of success: opening `/job/{id}/result` shows N ordered steps, each
with at least one image, and the images visibly correspond to the step
text. Quality of matching can be iterated on later.

---

End of plan.
