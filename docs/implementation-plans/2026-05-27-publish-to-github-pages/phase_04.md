# Publish to GitHub Pages — Phase 4: Docs + Operator Notes

**Goal:** Update `CLAUDE.md`, `README.md`, and `pyproject.toml` so future Claude sessions and operators can use the publish feature without reverse-engineering the code.

**Architecture:** Documentation only. No code changes.

**Tech Stack:** Markdown, pyproject.toml.

**Scope:** Phase 4 of 4 from `docs/design-plans/2026-05-27-publish-to-github-pages.md`.

**Codebase verified:** 2026-05-27

---

## Reference: verified codebase state

- `CLAUDE.md` section anchors:
  - "Project Structure" starts ~line 39, ends before "Conventions" (~line 77).
  - "Settings (Env Boundary)" at ~line 252.
  - "Invariants" at ~line 314.
  - "Gotchas" at ~line 329.
- `README.md` sections in order: How it works, Requirements, Install, Run, Deploy, Run on the cloud, Configuration: Modes, Cost expectations, Limitations, Troubleshooting, Approach-C quality escape hatch.
- `pyproject.toml` `markers` block at lines 13-15, currently just registers `cloud`.

---

<!-- START_TASK_1 -->
### Task 1: Update `CLAUDE.md` — module map, invariant, settings, gotchas

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Project Structure additions**

In the "Project Structure" bulleted list, after the `pipeline/` bullet (where `pipeline.py`, `audio.py` are mentioned), append a sub-entry describing the new modules:

Find this line:
```
- `pipeline/` — all video→steps stages. Pure logic in `captions.py`,
  `match.py`, parsing helpers; orchestration in `pipeline.py`. Audio
  extraction for the Whisper fallback lives in `audio.py`.
```

Append immediately after that bullet (same indentation):
```
  Static-bundle publishing splits along the same FCIS line: pure
  rendering + file-map composition in `publish.py`; the local-clone
  state, `gh`/`git` subprocesses, and asyncio serialization in
  `publish_repo.py`. The factory `build_publish_repo(settings)` is the
  only sanctioned construction site, mirroring `build_llm` /
  `build_embedder` / `build_whisper`.
```

In the `server.py` bullet, append a sentence:
```
  Adds POST `/job/{id}/publish` and `/job/{id}/unpublish` when
  `PUBLISH_ENABLED=1`; both swap the `_publish_controls_fragment.html`
  partial via HTMX.
```

**Step 2: Server Routes section**

In the "Server Routes" section, append two new bullets in the existing list (after the `frame` bullet):
```
- `POST /job/{id}/publish` → render a static bundle of the result page
  (via `pipeline.publish.build_static_bundle`), push it to the
  configured `PUBLISH_REPO` under `/<job_id>/` (via
  `pipeline.publish_repo.publish_job`), update the manifest's
  `published_url`/`published_at`, and return the
  `_publish_controls_fragment.html` fragment for HTMX swap. Only
  active when `PUBLISH_ENABLED=true`; 400 otherwise. 400 if status
  != `done`. 500 on `PublishError` (manifest is NOT updated).
- `POST /job/{id}/unpublish` → delete `<job_id>/` from the publish
  repo, clear the manifest's publish fields, return the fragment.
  No-op (still clears manifest, still returns fragment) if the job
  was never published.
```

**Step 3: Settings section additions**

After the Whisper settings block (the `WHISPER_MODEL` description), append a Publish settings block:
```

Publish-to-GitHub-Pages settings:
- `PUBLISH_ENABLED` (`publish_enabled: bool`, default `False`) — when
  `False`, the result page does not render the publish controls and
  the publish/unpublish routes 400. Operator opts in explicitly in
  `.env`.
- `PUBLISH_REPO` (`publish_repo: str`, default `santhony/vts-publish`) —
  `owner/name` of the shared GitHub repo that hosts every published
  job under `/<job_id>/`. First publish auto-creates this repo (public)
  if it doesn't exist; subsequent publishes reuse it.
- `PUBLISH_BRANCH` (`publish_branch: str`, default `main`) — the branch
  GitHub Pages is configured to serve.
- `PUBLISH_BASE_URL` (`publish_base_url: str`, default
  `https://santhony.github.io/vts-publish`) — the public URL prefix.
  Per-job URL is `<base>/<job_id>/`.
- `PUBLISH_CLONE_DIR` (`publish_clone_dir: Path`, default
  `data/publish_repo`) — local checkout reused across publishes to
  avoid the per-publish `git clone` cost.
```

**Step 4: Invariants — second-writer carve-out**

Update the existing line:
```
- Manifest writes only happen inside `pipeline.pipeline._update` (in
  the orchestrator) or directly in `server.py` for the initial
  `queued` state. Two writers, same atomic helper.
```

Replace with:
```
- Manifest writes happen in exactly three sanctioned sites: (1)
  `pipeline.pipeline._update` (the orchestrator), (2) the initial
  `queued` write in `server.py:POST /process`, and (3) the publish-
  state merges in `server.py:POST /job/{id}/publish` and
  `/job/{id}/unpublish`. All three use `write_json_atomic`. Do not
  add a fourth writer without revisiting this invariant.
```

**Step 5: Gotchas additions**

Append to the Gotchas section, after the existing "Pending operator items" bullet:
```
- **Publish requires `gh` + GitHub Pages**: the publish route shells
  out to `gh` and `git`; `gh auth login` (with `repo` scope) must be
  run on the host. First publish auto-creates `PUBLISH_REPO` and
  enables Pages via `gh api repos/<repo>/pages`. If your `gh` user
  lacks permission to create or push to the configured repo, the
  route 500s with the underlying `gh`/`git` stderr.
- **Public repo by design**: the default `vts-publish` is **public**,
  and job_ids leak the source YouTube id (`<video_id>_<6 hex>`).
  Operators handling private content should change `PUBLISH_REPO` to
  a private repo (note: GitHub Pages on private repos requires a paid
  plan).
- **Unpublish doesn't truly delete**: `git rm` + push leaves the
  removed files in commit history. The `<job_id>/` URL goes 404, but
  someone who knows the commit SHA can still recover the bundle.
  Truly removing requires `git filter-repo` + force-push (out of scope).
- **Single-process publish lock**: `PublishRepo` uses an
  `asyncio.Lock` to serialize publishes within one process; the
  single-process FastAPI deployment assumption already noted in this
  file applies here too. A multi-worker deployment would need an
  inter-process filesystem lock.
```

**Step 6: Verify**

Read the updated `CLAUDE.md` and confirm:
- Module map mentions `publish.py` + `publish_repo.py` with FCIS split.
- Server Routes lists the new POST endpoints.
- Settings section includes the 5 publish fields.
- Invariants references three writers (not two).
- Gotchas includes the four new bullets.

Run: `python -m pytest`
Expected: All tests pass (no code changed).

**Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "Document publish-to-github-pages feature in CLAUDE.md"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `README.md` — operator-facing Publish section

**Files:**
- Modify: `README.md` (insert a new `## Publish to GitHub Pages` section after `## Run`, before `## Deploy (single host)`)

**Step 1: Add the section**

Insert the following block immediately before the `## Deploy (single host)` heading (currently around line 117):

```markdown
## Publish to GitHub Pages

video-to-steps can snapshot a completed job's result page and push it
to a shared GitHub repository served via GitHub Pages, so the
step-by-step guide is shareable as a public URL.

### Prerequisites

- `gh` CLI installed and authenticated on the host that runs the
  server: `gh auth login` (select GitHub.com → HTTPS → authenticate
  with browser). The auth must include `repo` scope.
- `git` configured (`git config --global user.name "…" && git config
  --global user.email "…@…"`) so commits don't fail.
- Network access to github.com.

### Enable it

Set `PUBLISH_ENABLED=1` in `.env` (or your deployment env). Restart
the server. The result page (`/job/<id>/result`) now shows a "Publish
to GitHub Pages" button when the job is `done`.

### What happens on first publish

1. If `PUBLISH_REPO` (default `santhony/vts-publish`) doesn't exist
   under your authenticated `gh` user, it is created as a **public**
   repo.
2. GitHub Pages is enabled on `PUBLISH_BRANCH` (default `main`).
3. The result page is rendered as a self-contained HTML bundle
   (winner frames + a copy of `main.css`, paths rewritten to be
   relative) and pushed under `/<job_id>/`.
4. The button swaps to show the public URL and an "Unpublish" button.
5. Pages typically serves the new URL within ~30 seconds.

### Unpublishing

Click "Unpublish" on the result page. The `<job_id>/` directory is
removed and pushed; the URL goes 404 within ~30 seconds. **Note**:
`git rm` does not rewrite history — the bundle remains recoverable
from past commits. To remove a publish irrecoverably, do that
manually with `git filter-repo` + force-push.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `PUBLISH_ENABLED` | `false` | Master switch for the feature. |
| `PUBLISH_REPO` | `santhony/vts-publish` | `owner/name` of the shared publish repo. |
| `PUBLISH_BRANCH` | `main` | Branch Pages serves from. |
| `PUBLISH_BASE_URL` | `https://santhony.github.io/vts-publish` | Public URL prefix. |
| `PUBLISH_CLONE_DIR` | `data/publish_repo` | Local clone reused across publishes. |

### Verifying the path end-to-end

```bash
RUN_PUBLISH_SMOKE=1 python -m scripts.smoke_publish
```

Bundles a synthetic single-step job, pushes it, prints the URL, and
unpublishes when you press Enter. Useful as a one-shot operator check
of `gh` auth + Pages config.

### Privacy

The default repo is **public** and job_ids embed the source YouTube
video id. Publishing is equivalent to sharing the original video link
plus the generated guide. Use a private `PUBLISH_REPO` if that is
unacceptable (note: Pages on private repos requires a paid GitHub
plan).
```

**Step 2: Verify**

Read the updated section. Confirm prerequisites + env table render correctly in a Markdown preview if available.

Run: `python -m pytest`
Expected: All tests still pass.

**Step 3: Commit**

```bash
git add README.md
git commit -m "Document publish-to-github-pages operator workflow in README"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Register the `publish` pytest marker

**Files:**
- Modify: `pyproject.toml` (extend the `markers` list)

**Step 1: Edit `pyproject.toml`**

Update the existing `markers` block (currently lines 13-15):
```toml
markers = [
    "cloud: tests that call cloud APIs; skipped unless RUN_CLOUD_TESTS=1",
]
```

Replace with:
```toml
markers = [
    "cloud: tests that call cloud APIs; skipped unless RUN_CLOUD_TESTS=1",
    "publish: tests that hit the real publish path (gh + git + network); skipped unless RUN_PUBLISH_SMOKE=1",
]
```

**Step 2: Verify marker registration**

Run: `python -m pytest --markers | grep -E "(cloud|publish)"`
Expected: Both markers listed.

Run: `python -m pytest --strict-markers`
Expected: Suite passes (strict mode would error on undefined markers if anything uses `@pytest.mark.publish` without registration — currently nothing does, but this future-proofs Phase 2 Task 5's smoke script if anyone wraps it in a test).

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Register @pytest.mark.publish marker"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Update freshness date in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (top of file)

**Step 1: Update the freshness line**

Replace the existing `Last verified: 2026-05-20` near the top with today's date.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Bump CLAUDE.md freshness date"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Final verification

**Files:** none (verification only).

**Step 1: Run the full offline test suite**

Run: `python -m pytest -v`
Expected: All tests pass; new publish-related test counts visible:
- `tests/pipeline/test_publish_types.py`
- `tests/pipeline/test_publish.py`
- `tests/pipeline/test_storage.py`
- `tests/pipeline/test_publish_repo.py`
- `tests/test_config.py`
- `tests/test_server_publish.py`

**Step 2: Smoke the operator path (manual)**

With `gh auth login` complete and `.env` configured for a writable
`PUBLISH_REPO`:

```bash
RUN_PUBLISH_SMOKE=1 python -m scripts.smoke_publish
```

Expected: Published URL prints; opens correctly in a browser within
~30 seconds; pressing Enter unpublishes and the URL goes 404 within
~30 seconds.

**Step 3: Review docs for cross-references**

Skim `CLAUDE.md` and `README.md` and confirm every reference to a new
file, env var, or route resolves to something that actually exists in
the worktree.

**Step 4: Confirm git status is clean**

Run: `git status`
Expected: working tree clean, all commits on the
`publish-to-github-pages` branch.
<!-- END_TASK_5 -->
