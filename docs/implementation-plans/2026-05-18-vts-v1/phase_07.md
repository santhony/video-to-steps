# video-to-steps Implementation Plan — Phase 7: Hosting + docs

**Goal:** A fresh reader can go from `git clone` to a working hosted
instance using only the README. README replaces the pre-implementation
stub; Dockerfile gives a one-command deployment path; `.env.example` is
re-reviewed for clarity.

**Architecture:** No code changes. This phase ships docs and a thin Docker
wrapper around the work of Phases 1–6.

**Tech Stack:** `python:3.11-slim` Docker base, ffmpeg via apt-get, the
existing uv-managed venv approach inside the container.

**Scope:** 7 of 7 phases.

**Codebase verified:** 2026-05-18. All Phase 1–6 deliverables expected; this
phase touches only `README.md`, `.env.example`, and creates `Dockerfile`.

**External dependency findings:**
- ✓ `python:3.11-slim` includes `apt` so `apt-get install -y ffmpeg`
  works at image-build time. ~50 MB increase from ffmpeg + deps.
- ✓ `EXPOSE 8090` is informational; the actual binding is via
  `-p 8090:8090` at `docker run`. Setting `APP_HOST=0.0.0.0` inside the
  container is required for any port-publish to reach the app.
- ✓ Common reverse-proxy options for a single-host deploy: Caddy
  (simplest TLS + autorenewal), nginx (most familiar), or Tailscale
  Funnel (no public IP needed). README documents the conceptual
  expectation, not vendor-specific config.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC9: Deployment story
- **vts-v1.AC9.1 Success:** Default `APP_HOST=127.0.0.1`; setting `APP_HOST=0.0.0.0` is the only change required for cloud binding.
- **vts-v1.AC9.2 Success:** README documents reverse-proxy/VPN expectation for `0.0.0.0` deployments and warns against open-internet exposure without one.
- **vts-v1.AC9.3 Success:** README documents Mode A (MLX CLIP on Macbook) and Mode B (hybrid) with full env-var blocks; explicitly notes both are untested in v1.

---

<!-- START_TASK_1 -->
### Task 1: `Dockerfile`

**Verifies:** vts-v1.AC9.1 (default binding inside the image).

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Implementation:**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# System deps: ffmpeg is the only one not already in the slim base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (Dockerfile builds with the .dockerignore filter below).
COPY . .

# Default to 0.0.0.0 inside the container so a port publish actually works.
# A reverse proxy is still required for any public binding; see README.
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8090
EXPOSE 8090

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
```

`.dockerignore`:

```
.git
.worktrees
venv
data
__pycache__
*.pyc
.pytest_cache
.env
docs
tests
scripts
*.md
!README.md
```

**Verification:**

```bash
docker build -t vts-v1:dev .
docker run --rm -d -p 18090:8090 \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e JINA_API_KEY="$JINA_API_KEY" \
  -e VISION_API_KEY="$VISION_API_KEY" \
  --name vts-test vts-v1:dev
sleep 2
curl -sf http://127.0.0.1:18090/ | grep -c '<form'   # → 1
docker stop vts-test
```

Expected: image builds without errors; container starts; `GET /` returns
the form (grep prints 1). Containers without API keys still serve the
form — the keys are only needed when a job is submitted.

**Commit:**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(vts-v1): Dockerfile (python:3.11-slim + ffmpeg)"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: README rewrite

**Verifies:** vts-v1.AC9.2, vts-v1.AC9.3

**Files:**
- Modify: `README.md` (full rewrite — the current 14-line stub is
  replaced).

**Implementation:**

Below is the full README content. The structure is: pitch → install →
run-local → deploy → modes → troubleshooting → costs.

```markdown
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
```

**Verification:**

```bash
# Render the README into a Markdown previewer or just inspect length.
test -s README.md && wc -l README.md
grep -c "Mode A" README.md   # > 0
grep -c "Mode B" README.md   # > 0
grep -c "UNTESTED" README.md # ≥ 2 (Mode A + Mode B sections)
grep -c "127.0.0.1" README.md
grep -c "reverse proxy" README.md
```
Expected: all > 0; `UNTESTED` ≥ 2.

**Commit:**

```bash
git add README.md
git commit -m "docs(vts-v1): rewrite README — install/run/deploy/modes/troubleshooting"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `.env.example` final review

**Files:**
- Modify: `.env.example` (re-verify against final `Settings` field set;
  update comments if any drift).

**Implementation:**

After Phases 1–6 the `Settings` class is the authoritative variable list.
Diff `.env.example` against the actual `Settings` fields and ensure:

1. Every `Settings` field has a row in `.env.example`.
2. Mode A/B blocks contain a working set of overrides (commented).
3. Required-for-Mode-C variables (JINA_API_KEY, LLM_API_KEY, VISION_API_KEY)
   are flagged as required in their inline comments.

Use this audit command to catch missing rows:

```bash
python -c "
from config import Settings
fields = {f.alias or n.upper() for n, f in Settings.model_fields.items()}
print('In Settings but NOT in .env.example:')
import re
env = open('.env.example').read()
for k in sorted(fields):
    if not re.search(rf'^{re.escape(k)}=', env, re.MULTILINE):
        if not re.search(rf'^#\\s*{re.escape(k)}=', env, re.MULTILINE):
            print(' ', k)
"
```

Expected: empty list (every settings field is referenced).

If anything's missing, append it to the relevant mode block in
`.env.example`.

**Verification:**

```bash
source venv/bin/activate
python scripts/smoke_embed.py 2>/dev/null || true  # at least the env loads
grep -c "Mode C" .env.example   # ≥ 1
grep -c "Mode A" .env.example   # ≥ 1
grep -c "Mode B" .env.example   # ≥ 1
```
Expected: each mode is mentioned at least once.

**Commit:**

```bash
git add .env.example
git commit -m "docs(vts-v1): .env.example post-implementation audit"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Fresh-host rehearsal + run notes

**Verifies:** vts-v1.AC9.1, vts-v1.AC9.2, vts-v1.AC9.3 (operational
verification of the README itself).

**Files:**
- Create: `docs/implementation-plans/2026-05-18-vts-v1/rehearsal-notes.md`
- Possibly: small edits to README.md based on rehearsal findings.

**Implementation:**

This task is an operator action, not code. Stand up a fresh Linux host
(or a `python:3.11-slim` Docker container, or a fresh DigitalOcean droplet
— anywhere without prior state) and walk through the README top to
bottom. For each step record one of:

- ✓ worked as documented
- ✗ needed an extra step (and what the missing step was)
- ⚠ documented action took unexpected effort

Write the findings to `rehearsal-notes.md` and, where the README needs a
fix, patch it in a follow-up commit. The rehearsal-notes document is the
audit trail showing AC9 was actually verified.

**Rehearsal checklist (template content for `rehearsal-notes.md`):**

```markdown
# vts-v1 README rehearsal

**Host:** [VPS provider / OS / Python version]
**Date:** [YYYY-MM-DD]
**Rehearsed by:** [name]

## Walkthrough results

- [ ] `git clone` + `cd` — ✓ / ✗
- [ ] `cp .env.example .env` + fill in keys — ✓ / ✗
- [ ] `./setup.sh` — ✓ / ✗ (notes)
- [ ] `./start.sh` — ✓ / ✗
- [ ] `GET /` returns form — ✓ / ✗
- [ ] Submit a known short YouTube URL — ✓ / ✗
- [ ] Job page polls and reaches `done` — ✓ / ✗
- [ ] Result page shows ≥3 steps with frames — ✓ / ✗
- [ ] `./stop.sh` cleans up — ✓ / ✗
- [ ] `docker build -t vts-v1 .` succeeds — ✓ / ✗
- [ ] Docker container serves form on `localhost:8090` — ✓ / ✗
- [ ] `APP_HOST=0.0.0.0 ./start.sh` binds 0.0.0.0 — ✓ / ✗

## Findings

[List anything ✗ or ⚠ above with explanation and the fix to README.]

## AC9 verification

- AC9.1: APP_HOST default 127.0.0.1 (checked by reading config.py defaults
  and observing start.sh); changing to 0.0.0.0 worked without other
  changes — ✓ / ✗
- AC9.2: Reverse-proxy expectation called out in README — ✓ (line
  reference: ...)
- AC9.3: Mode A and Mode B documented with env-var blocks AND explicit
  UNTESTED note — ✓ (line references: ...)
```

**Verification:**

```bash
ls docs/implementation-plans/2026-05-18-vts-v1/rehearsal-notes.md
grep -c "AC9" docs/implementation-plans/2026-05-18-vts-v1/rehearsal-notes.md
```
Expected: file exists with at least one AC9 reference.

**Commit:**

```bash
git add docs/implementation-plans/2026-05-18-vts-v1/rehearsal-notes.md
# If README needed fixes, include them in the same commit:
git add README.md   # only if changed during rehearsal
git commit -m "docs(vts-v1): fresh-host rehearsal notes for README"
```

**Done when:** All boxes in the rehearsal checklist are ✓ (or each ✗ has
been fixed and re-rehearsed) and README review confirms AC9.1/AC9.2/AC9.3
are observably true.
<!-- END_TASK_4 -->
