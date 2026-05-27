# Publish to GitHub Pages Design

## Summary

Add a "Publish" / "Unpublish" pair of buttons to the result page that
snapshot a completed job's `/job/{id}/result` page into a self-contained
static bundle (HTML + referenced winner-frame JPGs + a copy of `main.css`)
and push it to a single shared GitHub repository `santhony/vts-publish`
under `/<job_id>/`. Pages serves the result at
`https://santhony.github.io/vts-publish/<job_id>/`.

The rendering reuses the existing `result.html` Jinja template via a
`static_mode=True` branch that emits relative paths instead of the live
server's `/job/{id}/...` URLs. Pure bundling logic lives in
`pipeline/publish.py` (Functional Core); the local clone, `gh repo create`
bootstrap, Pages enablement, and `git add/commit/push` live in
`pipeline/publish_repo.py` (Imperative Shell). Publish state is persisted
on the existing manifest (`published_url`, `published_at`) via
`write_json_atomic`, consistent with the project's
manifest-as-single-source-of-truth invariant.

## Definition of Done

Add a "Publish" / "Unpublish" pair of buttons to the result page. Publish
snapshots the existing `/job/{id}/result` page into a self-contained static
bundle (HTML + the referenced winner frames + a copy of `main.css`, with
paths rewritten to be relative) and pushes it to `santhony/vts-publish`
under `/<job_id>/` on the branch GitHub Pages is configured to serve. The
repo is auto-created on first publish if missing, with Pages enabled. The
published page is the result page verbatim (meta line included), no
internal status. Unpublish deletes the `<job_id>/` subdirectory, commits,
and pushes — URL goes 404. Publish state (`published_url`, `published_at`)
lives in `meta.json` via `write_json_atomic`. FCIS preserved:
bundling / path-rewriting is pure; `git` and `gh` calls live in the shell.

**Success:** From a `done` job, clicking Publish results in a public URL
like `https://santhony.github.io/vts-publish/<job_id>/` that renders the
full step-by-step guide with images, within ~30 seconds. Clicking
Unpublish makes that URL 404 within ~30 seconds. Manifest reflects current
state. Re-publishing an already-published job overwrites in place.

**Out of scope:** index page listing published jobs; editing/curation
before publish; custom domains; non-GitHub hosts; auth/access control on
published pages (they are public by design).

## Architecture

### High-level flow

```
[Result page]
  └── POST /job/{id}/publish
        └── pipeline.publish.build_static_bundle(...)        # pure
              → StaticBundle(html, files)
        └── pipeline.publish_repo.publish_job(job_id, ...)   # shell
              ├── ensure clone at data/publish_repo/
              ├── ensure remote repo (gh repo create) + Pages enabled
              ├── rsync bundle into <job_id>/
              ├── git add / commit / push
              └── return published_url
        └── _update(manifest, published_url, published_at)
        └── return HTMX fragment (button → "Unpublish" + URL)

[Result page]
  └── POST /job/{id}/unpublish
        └── pipeline.publish_repo.unpublish_job(job_id, ...) # shell
              ├── git rm -r <job_id>/
              ├── git commit / push
              └── return None
        └── _update(manifest, published_url=None, published_at=None)
        └── return HTMX fragment (button → "Publish")
```

### Functional Core: `pipeline/publish.py`

Pure module. Two main functions:

- `render_static_html(manifest, steps, frame_captions, step_links,
  templates_env) -> str` — renders `result.html` with `static_mode=True`.
  In static mode the template:
  - Drops the `<script src="/static/js/htmx.min.js">` tag (no HTMX needed
    on the snapshot).
  - Rewrites `href="/static/css/main.css"` → `href="main.css"`.
  - Rewrites `<a href="/">` brand link → `<a href="https://github.com/santhony/vts-publish">`
    (or omits the header entirely — TBD in implementation, leaning toward
    keeping a non-clickable brand to preserve layout).
  - Rewrites image `src` and `href` from `/job/{id}/frame/0001.jpg` →
    `frames/0001.jpg`.
  - Omits the publish-controls partial (only rendered in live mode).

- `build_static_bundle(manifest, steps, frame_captions, step_links,
  frames_dir, css_path, templates_env) -> StaticBundle` — composes
  `render_static_html` with the file map:

  ```python
  @dataclass(slots=True)
  class StaticBundle:
      html: str                       # contents of index.html
      file_map: dict[str, Path]       # relpath in bundle -> source on disk
      # e.g. "frames/0001.jpg" -> data/jobs/<id>/frames/0001.jpg
      #      "main.css"        -> static/css/main.css
  ```

  Only the union of frames referenced by `steps[*].frames` is included.
  Originals from `frames/`, not the embedding thumbnails.

The pure split lets us unit-test bundle composition (correct frames
included, correct relative paths, no leakage of absolute URLs) without
touching disk beyond reading source files.

### Imperative Shell: `pipeline/publish_repo.py`

Owns all state outside the functional core: the local clone, asyncio
serialization, and `gh`/`git` subprocess calls.

- `class PublishRepo` with a singleton-per-settings construction via
  `build_publish_repo(settings)` factory (mirroring `build_llm`, etc.).
- State: `Path` to local clone (`settings.publish_clone_dir`, default
  `data/publish_repo/`); `asyncio.Lock` to serialize concurrent publishes
  on a single process.
- `async def ensure_ready()` — idempotent bootstrap:
  1. If clone dir absent: `gh repo view <publish_repo>` → if 404,
     `gh repo create <publish_repo> --public --add-readme`.
  2. `gh api -X POST repos/<publish_repo>/pages -f
     "source[branch]=main" -f "source[path]=/"` (idempotent — 409 on
     already-enabled is fine).
  3. `git clone git@github.com:<publish_repo>.git <clone_dir>`.
  4. If clone present: `git fetch && git reset --hard origin/<branch>`.
- `async def publish_job(job_id, bundle) -> str`:
  1. Acquire lock.
  2. `ensure_ready()`.
  3. Pull latest.
  4. `rmtree(clone_dir / job_id)` if it exists; recreate.
  5. Write `bundle.html` to `<clone_dir>/<job_id>/index.html`.
  6. Copy each `bundle.file_map[k]` to `<clone_dir>/<job_id>/<k>`.
  7. `git add <job_id>/`; `git commit -m "publish <job_id>"`; `git push`.
  8. Return `f"{settings.publish_base_url}/{job_id}/"`.
- `async def unpublish_job(job_id) -> None`:
  1. Acquire lock.
  2. `ensure_ready()`.
  3. `git rm -r <job_id>/`; `git commit -m "unpublish <job_id>"`;
     `git push`.
- All subprocess calls go through `asyncio.create_subprocess_exec`,
  capturing stdout/stderr; non-zero exit → `PublishError(stderr)` which
  the server route catches and turns into an error fragment.

### Server changes (`server.py`)

- New routes:
  - `POST /job/{id}/publish` — validates job_id; loads manifest; rejects
    with 400 if not `done`; calls `build_static_bundle` + `publish_job`;
    updates manifest with `published_url` + `published_at`; returns
    `_publish_controls_fragment.html` for HTMX swap.
  - `POST /job/{id}/unpublish` — validates job_id; calls `unpublish_job`;
    clears manifest fields; returns the fragment.
- Manifest update goes through a small new helper analogous to
  `_update` but in the server (server already writes the initial `queued`
  manifest, so this is the second sanctioned writer — documented as a
  carve-out in CLAUDE.md, no third writer added).
- `_publish_controls_fragment.html` — new template fragment showing
  either:
  - "Publish to GitHub Pages" button (`hx-post`, `hx-indicator`,
    `hx-target=this`) when manifest has no `published_url`, or
  - the published URL + an "Unpublish" button when it does.
- `templates/result.html` — wrap the publish controls in a div
  `id="publish-controls"`, render the fragment via `{% include %}` in
  live mode, skip entirely in `static_mode`.

### Manifest schema additions

`pipeline.types.Manifest`:
```python
published_url: str | None = None
published_at: datetime | None = None
```
Both default to `None`. Existing manifests without these fields load fine
because `Manifest` uses `@dataclass(slots=True)` with defaults — no
migration needed. `_to_jsonable` already handles `datetime` and `None`.

### Settings additions (`config.py`)

```python
publish_repo: str = Field("santhony/vts-publish", alias="PUBLISH_REPO")
publish_branch: str = Field("main", alias="PUBLISH_BRANCH")
publish_base_url: str = Field(
    "https://santhony.github.io/vts-publish",
    alias="PUBLISH_BASE_URL",
)
publish_clone_dir: Path = Field(
    Path("data/publish_repo"), alias="PUBLISH_CLONE_DIR"
)
publish_enabled: bool = Field(False, alias="PUBLISH_ENABLED")
```

`publish_enabled=False` is the safer default — the button only renders
when the setting is true, so a developer cloning the repo doesn't
accidentally push to the maintainer's `vts-publish`. Operator opts in
explicitly in `.env`.

## Existing Patterns Followed

- **Functional Core / Imperative Shell.** Pure bundling in
  `pipeline/publish.py`; all `git`/`gh`/disk in `pipeline/publish_repo.py`.
  Docstring markers at the top of each module per house style.
- **Factory functions.** `build_publish_repo(settings)` joins
  `build_llm`/`build_vision`/`build_embedder`/`build_whisper` as the
  sanctioned construction site.
- **Atomic manifest writes.** Publish/unpublish go through
  `write_json_atomic`; manifest remains the single source of truth.
- **Dataclasses with `slots=True`.** `StaticBundle`,
  `PublishError`-payload — no Pydantic for internal types.
- **Test markers.** New `@pytest.mark.publish` (requires `gh` auth +
  network) for end-to-end tests, skipped by default. Pure
  `build_static_bundle` tests run in the default offline suite.
- **Settings env-alias contract.** Every new field uses
  `Field(alias="UPPER_SNAKE")` matching `.env.example`.

## Implementation Phases

### Phase 1 — Functional core + tests

- Add `StaticBundle` dataclass and `PublishError` to `pipeline/types.py`
  (or a new `pipeline/publish_types.py` if `types.py` is getting full).
- Add `pipeline/publish.py` with `render_static_html` +
  `build_static_bundle`.
- Add `templates/result.html` `static_mode` branches and the
  `_publish_controls_fragment.html` partial (controls only used in
  live mode, but the partial is needed for Phase 3).
- Unit tests in `tests/pipeline/test_publish.py`:
  - HTML contains rewritten relative frame paths, never absolute.
  - HTML contains `main.css` (not `/static/css/main.css`).
  - HTML omits the htmx script tag.
  - HTML omits the publish-controls block.
  - `file_map` contains exactly the union of frames in `steps[*].frames`,
    de-duplicated.
  - `file_map["main.css"]` resolves to the on-disk CSS.

### Phase 2 — Imperative shell + manifest fields

- Add `published_url`, `published_at` to `pipeline.types.Manifest`.
- Add `pipeline/publish_repo.py` with `PublishRepo` and
  `build_publish_repo`.
- Implement `ensure_ready`, `publish_job`, `unpublish_job` with
  asyncio.Lock serialization and subprocess error handling.
- Add settings fields in `config.py` + `.env.example`.
- Smoke script `scripts/smoke_publish.py` — bundles a fixture job and
  pushes to a throwaway repo (or runs against `vts-publish` with a
  `smoke-<timestamp>` job_id), guarded by `RUN_PUBLISH_SMOKE=1`.

### Phase 3 — Server routes + UI wiring

- Add `POST /job/{id}/publish` and `POST /job/{id}/unpublish` to
  `server.py`.
- Wire publish-controls partial into `result.html` (live mode only).
- Add `hx-indicator` spinner styling to `static/css/main.css`.
- Manifest update helper in `server.py` (second sanctioned writer,
  documented).
- Integration test `tests/test_server_publish.py` that mocks
  `build_publish_repo` to return a fake that records calls — verifies
  routes call the shell correctly and update the manifest.

### Phase 4 — Docs + operator notes

- Update `CLAUDE.md`:
  - New module map entries.
  - Document second-writer carve-out for manifest publish fields.
  - Add publish settings to the Settings section.
  - Add `gh` auth + Pages enablement to Gotchas.
- Update `README.md` with a "Publish" section: prerequisites
  (`gh auth login`, `repo` scope), enabling via `PUBLISH_ENABLED=1`,
  what happens on first publish, how to unpublish.
- Add `@pytest.mark.publish` marker registration and a one-line in the
  test-markers section.

## Additional Considerations

### Concurrency and partial-failure modes

- A `git push` that fails after a successful `git commit` leaves the
  local clone ahead of origin. `ensure_ready` always pulls before next
  publish; a subsequent retry succeeds. If push fails on the retry too,
  the server returns an error fragment with the stderr; manifest is
  **not** updated, so the UI still shows the Publish button.
- The asyncio lock prevents concurrent publishes on a single process. On
  a multi-worker deployment (not currently supported per CLAUDE.md
  "single-process FastAPI") we'd need a filesystem lock — explicitly
  noted as out of scope.

### Privacy

- The default repo (`vts-publish`) is **public**. The job_id format
  `<youtube_id>_<6 hex>` exposes the source YouTube video in the URL.
  Operators should be aware that publishing is equivalent to sharing the
  source link plus the generated guide.
- Unpublish removes the directory, but the commit history retains the
  removed files. A "truly delete" path would need
  `git filter-repo` + force-push, which is out of scope. The README
  section calls this out.

### Idempotency

- Re-publishing the same job_id overwrites in place (`rmtree` + rewrite).
- Re-unpublishing a job already absent from the repo no-ops in `git rm`
  (we check for the directory's existence first and return early if
  absent — the manifest is still cleared).
- `ensure_ready` is safe to call on every publish; the `gh repo view`
  cost is one API call.

### Why not GitHub Actions

- We considered pushing a "source of truth" JSON and letting a GitHub
  Action render to HTML. Rejected: adds a second deployment surface, a
  CI runtime cost, and a non-trivial latency between publish click and
  URL going live. Local render + direct push is simpler and faster.

## Glossary

- **Publish bundle / static bundle.** The output of
  `build_static_bundle`: a single HTML string plus a map of
  bundle-relative paths to source files on disk.
- **Publish repo.** The single shared GitHub repo (`santhony/vts-publish`
  by default) that hosts every published job at a `/<job_id>/` subpath.
- **Static mode.** A boolean flag passed into the Jinja `result.html`
  render that switches all internal URLs to bundle-relative form and
  drops the live-only HTMX / publish-controls bits.
- **Local clone.** The working-tree checkout of the publish repo
  maintained at `data/publish_repo/`, reused across publishes to avoid
  per-publish `git clone` cost.
- **Publish state.** The pair `(published_url, published_at)` on the
  manifest; presence indicates the job is currently published.
