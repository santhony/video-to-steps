# video-to-steps

Turn a YouTube instructional-video URL into an ordered, illustrated
step-by-step written guide.

Paste a link to a recipe, knot-tying tutorial, repair walkthrough, or
screencast; out comes a result page with N numbered steps, each one with
a polished imperative-mood instruction and one or more representative
frames from the source video.

## How it works

1. **Download** the video and its English auto-captions with `yt-dlp`.
2. **Parse** the VTT into time-coded cues and collapse YouTube's rolling
   repeats.
3. **Extract frames** at 1 fps with `ffmpeg`, then drop near-duplicates by
   perceptual-hash distance.
4. **Embed** every kept frame into a multimodal vector space (default:
   Jina v4).
5. **Outline** the transcript into 3–12 coarse steps via an LLM call.
6. **Match** the step briefs against the frame embeddings: per-step
   cosine top-k inside the step's time window.
7. **Caption** only the ~30 winning frames with a vision LLM (gpt-4o-mini
   default). This is the expensive call — we reserve it for frames that
   will appear on the result page.
8. **Refine** each step's text by feeding the brief, the matching cues,
   and the winning-frame captions back into the LLM, asking for 1–3
   second-person imperative sentences.
9. Render the result page.

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

## Configuration: Modes

Three deployment modes share the same code; the differences are
environment-only. Mode C is the only one verified in v1; Mode A and Mode
B are documented and code-pathed but **untested**.

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

### Mode A — Local (Macbook + qwen-studio + MLX CLIP). UNTESTED in v1.

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

**Caveats (v1):**
- The `MlxClipEmbedder` and qwen-studio code paths are not exercised by
  the v1 smoke test. Treat Mode A as "wired but not battle-tested."
- If `mlx_clip` fails to install, you can fall back to Mode C against a
  cloud Jina key with no code changes.
- The qwen-studio `/chat` SSE shape is auto-detected by `LLMClient`
  alongside the OpenAI shape; you don't need to change a flag.

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

## Troubleshooting

**"mlx_clip not installed" RuntimeError**
You set `EMBED_BACKEND=mlx_clip` on a non-Apple-Silicon host. Use
`EMBED_BACKEND=jina_v4` for Mode C or move to a Mac.

**"This video has no captions" error on result page**
The video has no auto-captions on YouTube. v1 fails fast in this case;
Whisper fallback is on the v2 roadmap.

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
