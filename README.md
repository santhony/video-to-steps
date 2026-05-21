---
title: video-to-steps
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8090
pinned: false
license: mit
short_description: YouTube instructional video → illustrated step-by-step guide
---

# video-to-steps

Turn a YouTube instructional-video URL into an ordered, illustrated
step-by-step written guide.

Paste a link to a recipe, knot-tying tutorial, repair walkthrough, or
screencast; out comes a result page with N numbered steps, each one with
a polished imperative-mood instruction and one or more representative
frames from the source video.

## How it works

1. **Download** the video and its English auto-captions with `yt-dlp`. If
   the video has no captions and `WHISPER_FALLBACK=1` is set, transcribe
   the audio locally with `faster-whisper` instead.
2. **Parse** the VTT into time-coded cues and collapse YouTube's
   rolling repeats — auto-captions emit the same line across many
   consecutive timestamped cues; we keep the first occurrence and drop
   any subsequent cue whose text is a prefix-or-equal of the previous
   one (`pipeline/captions.py:dedupe_rolling`).
3. **Extract frames** at 1 fps with `ffmpeg` (scaled to 720p), then drop
   near-duplicates by perceptual-hash distance (Hamming threshold 6).
4. **Embed** every kept frame into a multimodal vector space (default:
   Jina v4; 1024-d, L2-normalized).
5. **Outline** the transcript into 3–12 coarse steps via an LLM call
   (`prompts/outline.md`). The prompt explicitly excludes intros,
   outros, sponsor reads, and recap montages so the step list is
   actionable only.
6. **Match** the step briefs against the frame embeddings: per-step
   cosine top-3 inside the step's time window, padded ±2 s on each end.
   If the window is empty, fall back to the single frame nearest the
   step's midpoint.
7. **Caption** only the ~30 winning frames with a vision LLM
   (`gpt-4o-mini` default; `prompts/vision_caption.md`). This is the
   expensive call — we reserve it for frames that will appear on the
   result page.
8. **Refine** each step's text by feeding the brief, the matching cues,
   and the winning-frame captions back into the LLM
   (`prompts/refine.md`), asking for 1–3 second-person imperative
   sentences. `max_tokens=1500` to leave headroom for reasoning models'
   chain-of-thought.
9. Render the result page (each step deep-links back to its timestamp
   in the source YouTube video).

### Tunables

The defaults above sit in `config.py` (env-overridable) and the prompt
files in `prompts/`. The values most often worth changing:

| Knob | Default | Where |
|---|---|---|
| Frame extraction fps | 1 | `pipeline/frames.py:FixedFpsExtractor` |
| pHash dedup threshold | Hamming ≤ 6 | same |
| Step-to-frame top-k | 3 | `pipeline/match.py` |
| Step time window padding | ±2 s | `pipeline/match.py` |
| Outline LLM max_tokens | `LLM_MAX_TOKENS` (default 2048) | `.env`; raise for long transcripts |
| Refine LLM max_tokens | 1500 (hard-coded) | `pipeline/llm_refine.py` |
| Vision LLM max_tokens | 300 | `VISION_MAX_TOKENS` |
| LLM temperature | provider default (typically 1.0) | not overridden by this app |
| Whisper model | `base.en` | `WHISPER_MODEL` |

We do not set `temperature` or `seed` on outgoing LLM / vision calls
— provider defaults apply. That's why two runs on the same URL can
produce different step counts; if you need deterministic output for an
evaluation, fork `providers/llm.py` and add `temperature: 0, seed: N`
to the request body.

## Requirements

- Python 3.11+ (we use `uv` to fetch it if your host has only an older Python)
- `ffmpeg` on `PATH`
- `uv` (https://docs.astral.sh/uv/)
- API keys for the cloud providers (Mode C, default — see below)

## Install (local)

```bash
git clone https://github.com/santhony/video-to-steps.git
cd video-to-steps
cp .env.example .env
# Edit .env and fill in JINA_API_KEY, LLM_API_KEY, VISION_API_KEY.
./setup.sh
```

`setup.sh` creates a `venv/` with Python 3.11 (fetched by `uv` if needed)
and installs dependencies.

## Run

```bash
./start.sh
# server is now on http://127.0.0.1:8090
```

Open the URL in a browser, paste a short instructional YouTube link, and
wait. The job page polls itself every 2 seconds; once complete it shows a
"View result →" link.

To stop:

```bash
./stop.sh
```

## Deploy (single host)

The server binds `127.0.0.1` by default. **Do not expose the listener to
the public internet directly.** v1 has no rate limiting, no
authentication, and no abuse handling.

For a single-host deploy:

```bash
APP_HOST=0.0.0.0 ./start.sh
```

…and put a reverse proxy in front of the listener. Three reasonable
options:

- **Caddy** — easiest TLS; one Caddyfile line proxies to `127.0.0.1:8090`
  and Caddy handles certificates.
- **nginx** — most familiar; pair with `certbot` for TLS.
- **Tailscale Funnel** — no public IP needed; useful if you only want
  yourself + a small group to access the instance.

`APP_HOST=0.0.0.0` is the ONLY environment change required for the
cloud-binding step. The reverse-proxy step is your responsibility.

### Docker

```bash
docker build -t vts-v1 .
docker run -d --restart=unless-stopped \
  -p 127.0.0.1:8090:8090 \
  --env-file .env \
  -v $PWD/data:/app/data \
  --name vts vts-v1
```

The container defaults to `APP_HOST=0.0.0.0` internally; the `-p
127.0.0.1:8090:8090` ensures the published port is local-only. Add your
reverse proxy as above.

## Run on the cloud (Hugging Face Spaces)

The non-technical path: one OpenAI key, a free Hugging Face account, and
a Space duplication. The repository ships with the YAML header and
`Dockerfile` that Spaces needs — no local install required.

1. Sign in at <https://huggingface.co> and create an OpenAI API key at
   <https://platform.openai.com/api-keys>. You will be charged by OpenAI
   for each video processed; see "Cost expectations" below.
2. Open this repository on Hugging Face and click **Duplicate this
   Space** (or visit
   `https://huggingface.co/new-space?template=<your-username>/video-to-steps`).
   Choose the free CPU hardware tier.
3. In the new Space's **Settings → Variables and secrets**, add:
   - `OPENAI_API_KEY` (secret) — your OpenAI key.
   - `EMBED_BACKEND` = `vision_caption`
   - `LLM_BASE_URL` = `https://api.openai.com`
   - `LLM_API_KEY` = `${OPENAI_API_KEY}`
   - `LLM_MODEL` = `gpt-4o-mini`
   - `VISION_BASE_URL` = `https://api.openai.com`
   - `VISION_API_KEY` = `${OPENAI_API_KEY}`
   - `VISION_MODEL` = `gpt-4o-mini`
   - `TEXT_EMBED_MODEL` = `text-embedding-3-small`
   - `APP_BASIC_AUTH_USER` and `APP_BASIC_AUTH_PASS` — pick a username
     and a long random password. **Required.** Without these the Space
     URL is open to the public and anyone can spend your OpenAI key.
4. Restart the Space. Visit the Space's URL; your browser will prompt
   for the basic-auth credentials.

The Space sleeps when idle and wakes on the next request — the first
request after sleep takes ~30s while the container restarts. Cost per
video is whatever OpenAI bills you; expect a few cents per ~10 minute
video on `gpt-4o-mini` + `text-embedding-3-small`. The single-key mode
captions every kept frame instead of using a multimodal embedder, so
long videos cost more here than the Jina-based Mode C does locally.

## Configuration: Modes

Four deployment modes share the same code; the differences are
environment-only. Mode C is the v1 acceptance target. Mode D is the
single-key cloud deploy described above. Mode A has been
end-to-end smoke-tested on a single Apple Silicon machine (Qwen2.5-VL
via `mlx-vlm` for vision, qwen-studio/DS4 for chat, mlx_clip for
embeddings); Mode B (MLX CLIP locally + cloud LLM + cloud vision)
remains code-pathed but **untested**.

### Mode C — Cloud (default; the only mode tested in v1)

```bash
EMBED_BACKEND=jina_v4
JINA_API_KEY=...
JINA_MODEL=jina-embeddings-v4

LLM_BASE_URL=https://api.deepseek.com
LLM_PATH_CHAT=/v1/chat/completions
LLM_API_KEY=...
LLM_MODEL=deepseek-chat

VISION_BASE_URL=https://api.openai.com
VISION_PATH_CHAT=/v1/chat/completions
VISION_API_KEY=...
VISION_MODEL=gpt-4o-mini
```

Sub `gpt-4o-mini` with any OpenAI-shape vision endpoint;
sub the LLM with any OpenAI-shape chat endpoint (DeepSeek and Together
are tested shapes).

### Mode A — Local (Apple Silicon: mlx_clip + qwen-studio/DS4 + mlx-vlm)

Requires an Apple Silicon Mac, `mlx_clip`
(`pip install git+https://github.com/harperreed/mlx_clip`), and a
running [qwen-studio](https://github.com/santhony/qwen-studio) instance
on `127.0.0.1:8766`.

```bash
EMBED_BACKEND=mlx_clip
MLX_CLIP_MODEL=openai/clip-vit-base-patch32

LLM_BASE_URL=http://127.0.0.1:8766
LLM_PATH_CHAT=/chat
LLM_API_KEY=               # qwen-studio ignores; leave blank
LLM_MODEL=qwen2.5-coder

VISION_BASE_URL=http://127.0.0.1:8766
VISION_PATH_CHAT=/chat
VISION_API_KEY=
VISION_MODEL=qwen-vl
```

**Notes:**
- Smoke-tested on a 78-second knot-tying video and a 10-minute DIY
  woodworking video on a single Apple Silicon machine. "Tested" here
  means the pipeline completed end-to-end with sensible output, not
  that the matching is benchmarked.
- If `mlx_clip` fails to install on your host, fall back to Mode C
  against a cloud Jina key with no code changes.
- The qwen-studio `/chat` SSE shape is auto-detected by `LLMClient`
  alongside the OpenAI shape; you don't need to change a flag.
- For DS4-style reasoning models routed through qwen-studio, set
  `LLM_MAX_TOKENS` generously (we use 16000–80000 on long videos) —
  the CoT trace shares the token budget with the visible content.

### Mode D — Single-key (OpenAI only; cloud / Spaces deploy)

One key covers chat, vision, AND embeddings. `EMBED_BACKEND=vision_caption`
captions each kept frame with the vision LLM and embeds the caption
text, so no separate multimodal-embedding provider is needed.

```bash
EMBED_BACKEND=vision_caption
TEXT_EMBED_MODEL=text-embedding-3-small
# TEXT_EMBED_BASE_URL / TEXT_EMBED_API_KEY blank → reuse VISION_* (same key)

LLM_BASE_URL=https://api.openai.com
LLM_PATH_CHAT=/v1/chat/completions
LLM_API_KEY=${OPENAI_API_KEY}
LLM_MODEL=gpt-4o-mini

VISION_BASE_URL=https://api.openai.com
VISION_PATH_CHAT=/v1/chat/completions
VISION_API_KEY=${OPENAI_API_KEY}
VISION_MODEL=gpt-4o-mini
```

Trade-off: ~3–10× more vision-LLM calls than Mode C on long videos
because every kept frame is captioned. The simplest path for a
non-technical cloud user; see "Run on the cloud (Hugging Face Spaces)"
above for the click-through instructions.

### Mode B — Hybrid (MLX CLIP local + cloud LLM + cloud vision). UNTESTED in v1.

```bash
EMBED_BACKEND=mlx_clip
MLX_CLIP_MODEL=openai/clip-vit-base-patch32

LLM_BASE_URL=https://api.deepseek.com
LLM_PATH_CHAT=/v1/chat/completions
LLM_API_KEY=...
LLM_MODEL=deepseek-chat

VISION_BASE_URL=https://api.openai.com
VISION_PATH_CHAT=/v1/chat/completions
VISION_API_KEY=...
VISION_MODEL=gpt-4o-mini
```

There is no separate code path for Mode B — it's just Mode C with
`EMBED_BACKEND=mlx_clip`.

## Cost expectations

Per-job costs are recorded in `data/jobs/<id>/meta.json` and shown on the
result page. Order-of-magnitude estimates for a 3-minute video in Mode C
(prices as of May 2026):

| Stage                | Calls per job | Tokens per call | Cost (USD)  |
| ---                  | ---           | ---             | ---         |
| Frame embedding      | ~3            | ~5–20k          | $0.001–0.01 |
| Outline LLM          | 1             | ~3k             | <$0.001     |
| Step text embedding  | 1             | ~50             | <$0.0001    |
| Frame captioning     | ~30           | ~1k             | $0.01–0.05  |
| Step refine          | ~5–10         | ~1k             | $0.005–0.02 |
| **Total**            |               |                 | **~$0.02–0.10** |

A 30-minute video runs roughly 6× these numbers. Your mileage will vary
with provider pricing changes — check `pricing.py` and update entries
there when providers change rates.

## Limitations

This is a research-grade tool. The generated guides are useful, not
authoritative. Specifically:

- **Generated steps can be wrong.** The output is the product of an ASR
  pass, two LLM passes, and a similarity-based frame match — each one a
  source of error. Expect occasional missed steps, fused steps,
  hallucinated tool names, and frames that depict something adjacent
  rather than the action the text describes. Don't trust the output on
  anything safety-critical (medication, electrical work, food
  allergens) without verifying against the source video.

- **Non-deterministic.** Re-running the same URL through the same
  configuration will not produce the same step list, since the LLM
  passes are sampling at temperature > 0 internally. Two runs on the
  bowline tutorial in our testing produced 3 steps one time and 6
  steps another.

- **Captions are the input.** If YouTube's auto-captions are wrong,
  the steps will be wrong in the same way. Whisper fallback
  (`WHISPER_FALLBACK=1`, default off) avoids this by transcribing
  locally — but on first use it downloads the `faster-whisper` weights
  (~150 MB for `base.en`, larger for `small.en`/`medium.en`) and
  smaller Whisper models trade accuracy for speed.

- **Mode A and Mode B are not the v1 acceptance target.** Mode A has
  been smoke-tested on one machine; Mode B has never run. Quality and
  cost claims in this README are calibrated to Mode C.

- **Mode C has no built-in rate or cost ceiling.** A 30-minute video
  fans out ~30 vision-LLM calls per job (and more, if you raise top-k
  or frame fps). The caption stage uses an asyncio semaphore capped at
  16 concurrent (`CAPTION_MAX_IN_FLIGHT`) and retries 429s with
  exponential backoff, but there is no monthly cap or alert. Watch
  your provider dashboards.

- **The vision-caption assumption is unaudited.** In Mode C the entire
  step-to-frame match quality depends on the vision model producing
  consistent, useful descriptions of each frame. We have not
  systematically evaluated how different vision models (gpt-4o-mini
  vs. Qwen2.5-VL vs. Pixtral, etc.) trade off on instructional-video
  frames. Switching models is a config change, not a code change, so
  experimentation is cheap.

## Troubleshooting

**"mlx_clip not installed" RuntimeError**
You set `EMBED_BACKEND=mlx_clip` on a non-Apple-Silicon host. Use
`EMBED_BACKEND=jina_v4` for Mode C or move to a Mac.

**"This video has no captions" error on result page**
The video has no auto-captions on YouTube. Set `WHISPER_FALLBACK=1` in
`.env` to transcribe locally with `faster-whisper`. First run downloads
the model weights (~150 MB for `base.en`, the default); subsequent
runs reuse the cache. For better accuracy at higher CPU cost set
`WHISPER_MODEL=small.en` or `medium.en`.

**Vision model returns empty captions for some frames**
Some providers refuse certain images (people, certain content).
`caption_winners` tolerates per-frame failure: the affected frame's
caption is None and the refine pass uses only cue text for that step.
Manifest shows `status=done` regardless; no special UX.

**Empty content from DeepSeek v4**
DeepSeek v4 reasoning models can return empty `content` if
`LLM_MAX_TOKENS` is too low. Use `LLM_MAX_TOKENS=2048` or higher.

**ffmpeg not on PATH**
Install ffmpeg (`apt-get install ffmpeg`, `brew install ffmpeg`,
or use the provided Dockerfile).

## Approach-C quality escape hatch

If matching feels noisy (e.g., kitchen videos where every step looks
visually similar), the design supports a fallback `Embedder` that runs
the vision LLM on every frame and text-embeds the captions instead. This
trades cost for relevance. It's not implemented in v1 — write
`providers/embed_caption_then_text.py` and add it to the factory; no
other code changes needed.
