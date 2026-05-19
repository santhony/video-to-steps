# vts-v1 Test Requirements

**Source design:** `docs/design-plans/2026-05-18-vts-v1.md`
**Source implementation plan:** `docs/implementation-plans/2026-05-18-vts-v1/`
**Generated:** 2026-05-19

## Overview

vts-v1 is a FastAPI + HTMX app that converts a YouTube URL into per-step
captioned frames via a Mode-C pipeline (Jina embedder + cloud LLM + cloud
vision). This document is the final acceptance gate after implementation
is complete: a checklist that maps each Acceptance Criterion (AC) from
the design plan to either an automated test, a manual verification
procedure, or both. Two consumers use it: the test-analyst agent (to
confirm automated coverage exists) and a human reviewer (to perform the
manual steps that automation cannot judge — e.g. "frames are visibly
relevant" or "caption focuses on actions, tools, and materials").

How to use:

1. Read the "Automated test coverage" table and confirm each row's test
   file exists and passes.
2. Run the "Manual verification steps" for ACs marked `partial` or
   `manual only`.
3. Run the end-to-end smoke test once on a known short YouTube video and
   verify the listed observations at each stage.
4. Only mark vts-v1 done when every AC sub-case has been satisfied.

Test conventions:

- Default pytest run (`pytest`) skips `@pytest.mark.cloud` tests. Cloud
  tests require `RUN_CLOUD_TESTS=1` and a configured cloud LLM/vision
  endpoint.
- Provider stubs (qwen-studio shape, OpenAI shape) are exercised
  in-process via httpx mock transports — no live network for unit tests.
- Frame extractor tests synthesize tiny ffmpeg-generated mp4s into
  `tmp_path` so nothing binary is checked in.
- The single checked-in image fixture is
  `tests/providers/fixtures/test_frame.jpg`.

---

## Automated test coverage

### vts-v1.AC1 — End-to-end Mode C pipeline produces a usable result

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC1.1 | A YouTube URL of a short instructional video produces a `steps.json` with ≥3 ordered steps, each with ≥1 frame. | partial | `tests/pipeline/test_pipeline.py` (cloud-marked end-to-end); manual smoke confirms YouTube path | Cloud-marked `run_job` end-to-end produces a `steps.json` with ≥3 ordered steps, each with ≥1 frame. Unit-level orchestrator test uses stub providers + a short fixture mp4 to confirm the same invariants without network. |
| vts-v1.AC1.2 | Each step's `instruction` field is 1–3 second-person imperative sentences. | partial | `tests/pipeline/test_llm_refine.py::test_refine_produces_imperative_sentences` | Asserts sentence count is 1–3 and each begins with an imperative verb (lowercased token in a small allowlist or starts-with-verb heuristic). Style fit ("genuinely sounds like an instruction") is reviewed manually. |
| vts-v1.AC1.3 | Each step's frames are visibly relevant to the instruction text. | manual only | see "Manual verification steps" §M1 | Pure human judgment on the checked-in test video. |

### vts-v1.AC2 — YouTube download and caption parsing

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC2.1 | yt-dlp downloads MP4 at ≤720p and the English VTT auto-caption track. | partial | `scripts/smoke_phase3.py` (manual); `tests/pipeline/test_pipeline.py` (cloud) | Unit tests mock yt-dlp; the real network behavior is exercised by `smoke_phase3.py` and the cloud-marked pipeline test. Verify the downloaded MP4 height ≤720 and a `.en.vtt` (or equivalent English VTT) sibling exists. |
| vts-v1.AC2.2 | VTT parsing returns a `list[Cue]` in temporal order. | yes | `tests/pipeline/test_captions.py::test_parse_vtt` | Parses `tests/pipeline/fixtures/sample.vtt`, asserts result is a `list[Cue]`, all `start <= end`, and starts are non-decreasing. |
| vts-v1.AC2.3 | `dedupe_rolling` collapses YouTube auto-caption rolling-repeat patterns into single Cues. | yes | `tests/pipeline/test_captions.py` (rolling dedupe cases) | Feeds rolling-repeat fixture cues; asserts post-dedupe sequence is strictly shorter and contains no consecutive prefix-overlap pairs. |
| vts-v1.AC2.4 | When yt-dlp returns no captions and `WHISPER_FALLBACK` is disabled, the orchestrator writes `status="error"` with a clear message in `manifest.error`. | yes | `tests/pipeline/test_pipeline.py` (no-caption stub path) | Stubs `download_video_and_captions` to return `(video, None)` with `WHISPER_FALLBACK=False`; asserts the on-disk `meta.json` shows `status=="error"` and `error` contains a human-readable string mentioning captions. |

### vts-v1.AC3 — Frame extraction with deduplication

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC3.1 | `FixedFpsExtractor(fps=1.0, dedup=False)` extracts one frame per source second at ≤720p height. | yes | `tests/pipeline/test_frames.py::test_fixed_fps_extracts_one_per_second_at_720_or_less` | Generates a static N-second mp4 with ffmpeg; asserts extracted count ≈ N (±1 boundary) and max image height ≤ 720. |
| vts-v1.AC3.2 | `dedup=True, hamming_max=6` produces strictly fewer frames on cut-heavy video and equal-or-fewer on any input. | yes | `tests/pipeline/test_frames.py::test_dedup_strictly_reduces_on_cut_heavy_video` and `::test_dedup_equal_or_fewer_on_any_input` | First test: cuts-heavy synthetic mp4 → asserts `len(dedup_on) < len(dedup_off)`. Second test: any input → asserts `len(dedup_on) <= len(dedup_off)`. |
| vts-v1.AC3.3 | pHash dedup never drops the first frame. | yes | `tests/pipeline/test_frames.py::test_phash_filter_never_drops_first` | Even with `hamming_max=0` (pathological), the first frame is always preserved. |

### vts-v1.AC4 — Provider abstractions are config-only switchable

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC4.1 | `LLMClient` chats against both qwen-studio `/chat` and OpenAI-shape `/v1/chat/completions`, distinguishing by response-shape auto-detect. | yes | `tests/providers/test_llm_client.py` | Two mock transports: one returns raw `data: <text>` SSE chunks (qwen-studio), one returns `chat.completion` JSON (OpenAI). Asserts both produce the same `ChatResponse.text` and that auto-detect picks the correct parser. |
| vts-v1.AC4.2 | `LLMClient.chat()` strips `<think>...</think>` blocks. | yes | `tests/providers/test_llm_client.py` (think-strip case) | Mocked response contains a `<think>chain of thought</think>real answer` payload; asserts returned `.text == "real answer"` (no `<think>` substring). |
| vts-v1.AC4.3 | `JinaEmbedder.embed_images()` and `.embed_texts()` return L2-normalized float32 vectors of consistent dimensionality. | yes | `tests/providers/test_jina_embedder.py` | Asserts dtype is `float32`, `np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)` for every row, and that image and text embedding dims match. |
| vts-v1.AC4.4 | Switching `EMBED_BACKEND` from `jina_v4` to `mlx_clip` requires only an env-var change; `mlx_clip` raises a clear actionable error without the dep installed. | yes | `tests/providers/test_embed_factory.py` and `tests/providers/test_mlx_clip_factory.py` | Factory test: setting `EMBED_BACKEND=jina_v4` yields a `JinaEmbedder`; setting `EMBED_BACKEND=mlx_clip` on a host without `mlx_clip` raises `ImportError`/`RuntimeError` whose message names the missing package and `pip install` hint. |
| vts-v1.AC4.5 | `VisionCaptioner.caption()` returns a 1–2 sentence caption focused on actions, tools, and materials for a checked-in test still frame. | partial | `tests/providers/test_vision_captioner.py` (mocked shape); manual §M2 | Unit test mocks an OpenAI vision response and asserts caller sends the prompt template containing the words "action", "tool", "material" (or per prompts/ template) and that the returned text is 1–2 sentences. Content quality (actually focused on actions/tools/materials) is judged manually against `tests/providers/fixtures/test_frame.jpg`. |

### vts-v1.AC5 — LLM outline and refine passes

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC5.1 | `llm_outline` returns ≥3 `StepOutline`s with non-overlapping time ranges that together cover the input transcript span. | yes | `tests/pipeline/test_llm_outline.py::test_outline_well_formed_json_object_with_steps_key` (and siblings) | Mocked LLM returns a well-formed outline; assertions: `len(out) >= 3`; pairwise non-overlap (`out[i].end <= out[i+1].start`); `out[0].start <= transcript_start` and `out[-1].end >= transcript_end` (or coverage within configured pad). |
| vts-v1.AC5.2 | `llm_outline` parses correctly when `response_format={"type":"json_object"}` is supported. | yes | `tests/pipeline/test_llm_outline.py::test_outline_well_formed_json_object_with_steps_key` | Mocked response is a JSON object with a top-level `"steps"` key; asserts the parser unwraps it correctly. |
| vts-v1.AC5.3 | Slice-fallback parses a fixture containing prose around the JSON `[…]` block. | yes | `tests/pipeline/test_llm_outline.py::test_outline_slice_fallback_with_prose` and `::test_slice_finds_balanced_brackets_inside_strings` and `::test_parse_outline_raises_on_unrecoverable` | Prose-around-JSON case: parser recovers the array. Balanced-brackets case: `[`/`]` inside string literals don't confuse the slicer. Unrecoverable case: `ValueError` raised. |
| vts-v1.AC5.4 | `llm_refine` produces 1–3 second-person imperative sentences per step, incorporating winning-frame captions. | yes | `tests/pipeline/test_llm_refine.py::test_refine_produces_imperative_sentences` and `::test_refine_parses_object_with_prose_around` and `::test_refine_falls_back_when_chat_fails` | Imperative-sentence shape (count + first-token check); prompt-includes-captions check via mock-call inspection; fallback to outline text when chat fails. |

### vts-v1.AC6 — Frame-to-step matching

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC6.1 | `match` restricts candidate frames per step to `[step.start - pad, step.end + pad]` before scoring. | yes | `tests/pipeline/test_match.py` (window restriction case) | Constructs synthetic frames spanning a wide range; only frames within the padded window become candidates (rest are excluded even if their embeddings would score higher). |
| vts-v1.AC6.2 | `match` picks the top-k frames by cosine via `frame_emb @ step_emb` within the candidate window. | yes | `tests/pipeline/test_match.py` (top-k cosine case) | Builds candidate frames with hand-chosen embeddings; asserts the top-k selection matches the analytic dot-product ordering. |
| vts-v1.AC6.3 | When the candidate window is empty, `match` falls back to the single nearest frame to step midpoint. | yes | `tests/pipeline/test_match.py` (empty-window fallback case) | Step `[start, end]` falls in a gap between extracted frames; asserts exactly one frame is returned and it is the temporal nearest to `(start+end)/2`. |

### vts-v1.AC7 — Cost tracking and manifest

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC7.1 | At the end of a successful job, `meta.json.cost.total_usd` is non-zero and equals the sum of per-call costs from `pricing.py`. | yes | `tests/pipeline/test_pipeline.py` (orchestrator happy path with stubs) | End-to-end stubbed run; reads final `meta.json`; asserts `cost.total_usd > 0` and equals the sum of `cost.chat_usd + cost.embed_usd + cost.vision_usd` computed via `pricing.py` helpers given the recorded token counts. |
| vts-v1.AC7.2 | `meta.json` is atomically written; the status-poll endpoint always returns parseable JSON. | yes | Phase 1 inline `python -c` smoke (see phase_01.md Task 5) + `tests/test_server.py::test_status_fragment_running` | Phase 1 smoke asserts `write_json_atomic` produces a single readable JSON file after replace. Server test repeatedly polls `/job/{id}/status` while the orchestrator stub writes; every response parses without exception. |
| vts-v1.AC7.3 | A configured model not present in `pricing.py` records zero for that line item and logs a startup warning; the job still completes. | yes | `pricing.py` inline doctest-style assertions (phase_01.md Task 6) + see phase_05.md Task with unknown-model stub | `compute_chat_cost('unknown-model-xyz', 1_000_000, 1_000_000) == 0.0` and the module logs a warning at first miss. Orchestrator test confirms job ends with `status=done` despite the unknown model. |

### vts-v1.AC8 — Server and result page

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC8.1 | `GET /` renders the URL input form. | yes | `tests/test_server.py::test_index_renders_form` | Response 200; body contains `<form` and an input named for the URL. |
| vts-v1.AC8.2 | `POST /process` with a valid YouTube URL returns 303 redirect to `/job/{id}` and spawns a pipeline task. | yes | `tests/test_server.py::test_process_redirects_303_and_writes_manifest` and `::test_process_accepts_known_youtube_shapes` | Status code is 303, `Location` matches `/job/<uuid>`, and a fresh `meta.json` is written. Known YouTube URL shapes (`youtube.com/watch`, `youtu.be/...`, `youtube.com/shorts/...`) all accepted. |
| vts-v1.AC8.3 | `POST /process` with a non-YouTube URL returns 400 with a human-readable message. | yes | `tests/test_server.py::test_process_rejects_non_youtube_400` | Status code is 400; response body contains a non-empty message string. |
| vts-v1.AC8.4 | `/job/{id}/status` returns an HTMX fragment showing current `status`, `progress`, and running `cost.total_usd`. | yes | `tests/test_server.py::test_status_fragment_running` and `::test_status_fragment_done_stops_polling` | Running case: fragment contains `status`, the current `progress` string, and `cost.total_usd` rendered as currency. Done case: fragment also signals HTMX to stop polling (e.g. no `hx-get` attr, or `HX-Trigger: jobDone` header). |
| vts-v1.AC8.5 | `/job/{id}/result` renders ordered steps with caption-alt-texted thumbnails and small print of `mode`, `cost.total_usd`, embedder, vision model, LLM model. | yes | `tests/test_server.py::test_result_page_renders_steps_and_meta` and `::test_result_redirects_when_not_done` and `::test_unknown_job_id_404` | Renders steps in order; every `<img>` has a non-empty `alt`; small-print region contains `mode`, `total_usd`, `embed_backend`, `vision_model`, `llm_model` substrings. Not-done jobs redirect to status page; unknown id returns 404. |

### vts-v1.AC9 — Deployment story

| AC | AC text | Automated? | Test location | What the test verifies |
|---|---|---|---|---|
| vts-v1.AC9.1 | Default `APP_HOST=127.0.0.1`; setting `APP_HOST=0.0.0.0` is the only change required for cloud binding. | yes | inline `grep` checks in phase_07.md Task verification + `tests/test_server.py` (settings load) | `app/settings.py` default for `app_host` is `"127.0.0.1"`. Phase 7 verification greps `.env.example` to confirm `APP_HOST=127.0.0.1` is the documented default. |
| vts-v1.AC9.2 | README documents reverse-proxy/VPN expectation for `0.0.0.0` deployments and warns against open-internet exposure without one. | partial | inline `grep` checks in phase_07.md Task 2; manual §M3 | Phase 7 grep: README contains the strings "reverse proxy" and "0.0.0.0" near a warning section. Substantive accuracy of the warning is reviewed manually. |
| vts-v1.AC9.3 | README documents Mode A (MLX CLIP on Macbook) and Mode B (hybrid) with full env-var blocks; explicitly notes both are untested in v1. | yes | inline `grep` checks in phase_07.md Task 2 | `grep -c "Mode A" README.md > 0`, `grep -c "Mode B" README.md > 0`, `grep -c "UNTESTED" README.md >= 2`. |

---

## Manual verification steps

Each step lists: setup → action → expected observation → pass/fail.

### M1 — vts-v1.AC1.3: frames are visibly relevant to instructions

**Setup**
- Identify (or check in) a known short instructional YouTube video used
  for v1 acceptance (e.g. ≤5 min "how to make X" with clear visual
  steps). Note the URL.
- Configure Mode C env vars (cloud LLM + Jina + cloud vision) and start
  the server: `./start.sh`.

**Action**
1. Open `http://127.0.0.1:8090/` in a browser.
2. Paste the test-video URL into the form, submit.
3. Wait for the status page to show `status=done`.
4. Click "View result".

**Expected observation**
- Result page lists ordered steps.
- For each step, the displayed thumbnail(s) plausibly depict the action,
  tool, or material named in the instruction text. (Imperfect matches
  are acceptable; gross mismatches — e.g. a person's face for a "stir
  the mixture" step — are not.)
- At least 2 of every 3 steps reviewed should be a visible match.

**Pass criteria**
- ≥ 80% of steps have at least one visibly relevant frame.
- No step has a frame that contradicts the instruction (e.g. "raw
  ingredients" frame on a "serve the finished dish" step).

### M2 — vts-v1.AC4.5: vision caption focuses on actions, tools, materials

**Setup**
- Ensure `tests/providers/fixtures/test_frame.jpg` is present.
- Configure cloud vision env vars.

**Action**
- Run `python scripts/smoke_vision.py` (or equivalent ad-hoc invocation
  that calls `VisionCaptioner.caption()` on the fixture).

**Expected observation**
- Returned text is 1 or 2 complete sentences.
- Content mentions at least one of: an action (verb describing what is
  happening), a tool (named object being used), a material (named
  ingredient or substance). Generic scene descriptions ("a kitchen with
  good lighting") are a fail.

**Pass criteria**
- Sentence count ∈ {1, 2}.
- Reviewer can underline at least one action, tool, OR material noun in
  the caption.

### M3 — vts-v1.AC9.2: README deployment warning is substantive

**Setup**
- Open the final `README.md` in any Markdown previewer.

**Action**
- Locate the "Deployment" / "Cloud binding" section.

**Expected observation**
- The section explicitly states that `APP_HOST=0.0.0.0` exposes the
  server, that a reverse proxy (or VPN/Tailscale) is required, and that
  no auth is built into vts-v1.
- The warning is hard to miss — not buried in a parenthetical.

**Pass criteria**
- A new reader who skims only the deployment section understands they
  must put something in front of vts-v1 before binding `0.0.0.0`.

### M4 — vts-v1.AC1.2 wording quality (supplement to AC1.2 automation)

**Setup**
- Same as M1.

**Action**
- Read each step's `instruction` text on the result page.

**Expected observation**
- Each instruction starts with an imperative verb ("Add", "Heat",
  "Whisk", "Place", "Press", "Connect", etc.).
- No instruction is phrased in third person ("The chef adds the eggs")
  or as a description ("Eggs are added").
- Length per step is 1–3 sentences.

**Pass criteria**
- 100% of steps obey the imperative-second-person rule. Any failure is
  a fail (this is a hard contract).

---

## End-to-end smoke test

A single golden-path scenario walking from `POST /process` to
`/job/{id}/result` on a known short YouTube video.

**Preconditions**
- Mode C env vars configured: `LLM_BASE`, `LLM_MODEL`, `VISION_BASE`,
  `VISION_MODEL`, `JINA_API_KEY` (or local Jina), `EMBED_BACKEND=jina_v4`.
- `APP_HOST=127.0.0.1`, `APP_PORT=8090`.
- A working internet connection (yt-dlp + cloud providers).
- Server started via `./start.sh`.
- A known short instructional YouTube URL (the v1 reference video).

**Walkthrough**

1. **GET /** — open `http://127.0.0.1:8090/` in a browser.
   - Observe: form with a single URL input and a submit button.
   - Expected: status 200, no console errors.

2. **POST /process** — paste the URL and submit.
   - Observe: browser navigates to `/job/<uuid>` (303 redirect under
     the hood; visible as the new URL).
   - Expected: status page shows `status=queued` or `status=running`.

3. **Status polling** — leave the status page open.
   - Observe: progress text advances through stages ("downloading
     video" → "parsing captions" → "extracting frames" → "embedding
     frames" → "outlining steps" → "embedding step briefs" → "matching
     frames to steps" → "captioning winners" → "refining
     instructions").
   - Observe: running cost increments visibly across LLM/embed/vision
     stages.
   - Expected: no torn JSON, no 5xx in HTMX fragments, polling stops
     when status reaches `done`.

4. **GET /job/{id}/result** — click the "View result" link once
   `status=done`.
   - Observe: page renders ≥3 ordered steps.
   - Observe: each step shows its instruction text (1–3 imperative
     sentences) and at least one thumbnail with non-empty `alt`.
   - Observe: small print at top or bottom shows mode (`C`), final
     `total_usd`, embedder (`jina_v4`), vision model, LLM model.
   - Expected: page is fully renderable; clicking a thumbnail opens
     the full-size frame jpg under `/job/{id}/frame/NNNN.jpg`.

5. **Disk artifacts** — inspect `<jobs_root>/<job_id>/` on the server.
   - Observe: `meta.json` (final manifest), `steps.json`, `outline.json`,
     `frame_captions.json`, `video.mp4`, captions `.vtt`, and a
     `frames/` directory of `NNNN.jpg` files.
   - Expected: `meta.json.status == "done"`,
     `meta.json.cost.total_usd > 0`, `len(steps.json) >= 3`, each step
     has `frames` array of length ≥ 1.

6. **Atomicity sanity** — while a job is `running`, repeatedly `curl
   http://127.0.0.1:8090/job/<id>/status` in a tight loop.
   - Observe: every response body parses as HTML (no half-written JSON
     anywhere underneath).
   - Expected: zero parse errors over 100 consecutive requests.

**Smoke-test pass criteria** — all six observations succeed in a single
run. If any step fails, the corresponding AC sub-case is also marked
fail.

---

## Coverage summary

| AC group | Sub-cases | Fully automated | Partial | Manual only |
|---|---:|---:|---:|---:|
| AC1 end-to-end pipeline | 3 | 0 | 2 | 1 |
| AC2 download + captions | 4 | 3 | 1 | 0 |
| AC3 frame extraction | 3 | 3 | 0 | 0 |
| AC4 provider abstractions | 5 | 4 | 1 | 0 |
| AC5 outline + refine | 4 | 4 | 0 | 0 |
| AC6 matching | 3 | 3 | 0 | 0 |
| AC7 cost + manifest | 3 | 3 | 0 | 0 |
| AC8 server + result page | 5 | 5 | 0 | 0 |
| AC9 deployment | 3 | 2 | 1 | 0 |
| **Total** | **33** | **27** | **5** | **1** |

When the test-analyst agent runs, it should confirm all 27
fully-automated AC sub-cases have a passing test at the listed
locations. The 5 partial cases need both the automated assertion and
the matching manual step. The 1 manual-only case (AC1.3) is checked at
the end via the M1 procedure.
