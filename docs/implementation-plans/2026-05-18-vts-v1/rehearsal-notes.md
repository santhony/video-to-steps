# vts-v1 README rehearsal

**Environment:** CI-like automated session (not a real fresh host)
**Date:** 2026-05-19
**Rehearsed by:** Claude Code (structural verification)

## Structural verification results

The following were verified in the current environment without a real fresh-host walkthrough:

### Shell script syntax
- ✓ `bash -n start.sh` — valid syntax
- ✓ `bash -n stop.sh` — valid syntax
- ✓ `bash -n setup.sh` — valid syntax

### Application state
- ✓ `import server` → `server.app.title` = 'video-to-steps'
- ✓ Default `APP_HOST` = '127.0.0.1' (read from `config.get_settings()`)
- ✓ README contains reverse-proxy guidance (lines 80, 105)
- ✓ README documents Mode A + UNTESTED note (line 135)
- ✓ README documents Mode B + UNTESTED note (line 165)

### Docker support
Docker command not available in this environment. Dockerfile syntax verified manually:
- ✓ `Dockerfile` present, 24 lines
- ✓ Uses `FROM python:3.11-slim`
- ✓ Installs `ffmpeg` via `apt-get`
- ✓ Sets `ENV APP_HOST=0.0.0.0`
- ✓ `CMD` runs `uvicorn server:app --host 0.0.0.0 --port 8090`
- ✓ `.dockerignore` present with expected filters

**Note:** Real Docker build (`docker build -t vts-v1:dev .`) must be verified by operator on a host with Docker available.

## Pending items (require real fresh-host operator action)

The following items **cannot** be verified in an automated CI environment and **must** be completed by an operator on a real fresh host:

- [ ] `git clone https://github.com/santhony/video-to-steps.git` and `cd` into it
- [ ] `cp .env.example .env` and populate JINA_API_KEY, LLM_API_KEY, VISION_API_KEY
- [ ] `./setup.sh` completes without errors
- [ ] `./start.sh` starts the server without errors
- [ ] `GET http://127.0.0.1:8090/` returns HTML form (at least one `<form` tag)
- [ ] Submit a known short YouTube URL (must have auto-captions)
- [ ] Job page polls and reaches `done` status
- [ ] Result page displays ≥3 steps with frames
- [ ] `./stop.sh` stops the server cleanly
- [ ] `docker build -t vts-v1 .` builds without errors (operator must have Docker)
- [ ] Docker container serves form on `localhost:8090`
- [ ] `APP_HOST=0.0.0.0 ./start.sh` binds to 0.0.0.0 (verify with `netstat` or `lsof -i :8090`)

## AC9 verification summary

### AC9.1: Default binding + cloud setting
- **Verified:** Default `APP_HOST` is '127.0.0.1' (line 7 of `.env.example`, confirmed via `config.get_settings()`)
- **Verified:** Setting `APP_HOST=0.0.0.0` is the only environment change required (documented in README line 78)
- **Status:** Structurally ✓; operator must verify actual network binding on fresh host

### AC9.2: Reverse-proxy expectation documented
- **Location:** README.md lines 80–105
- **Content:**
  - Line 75: "Do not expose the listener to the public internet directly."
  - Lines 80–84: Three reverse-proxy options (Caddy, nginx, Tailscale Funnel)
  - Line 101: Explicit note that reverse proxy is operator's responsibility
  - Docker section (lines 96–105): `-p 127.0.0.1:8090:8090` enforces local-only binding
- **Status:** ✓ Documented as required

### AC9.3: Mode A and Mode B documented with UNTESTED note
- **Mode A:** README.md line 135
  - Full env-var block: lines 144–156 (commented, ready to copy)
  - Explicit UNTESTED note: "UNTESTED in v1"
  - Caveats section: lines 157–162
- **Mode B:** README.md line 165
  - Full env-var block: lines 169–180
  - Explicit UNTESTED note: "UNTESTED in v1"
  - Explanation: "No separate code path for Mode B" (lines 182–183)
- **Status:** ✓ Documented and flagged as untested

## Findings

No AC9-blocking issues found in structural verification. All documentation in place:
- README is complete and comprehensive
- `.env.example` covers all Settings fields (audit passed)
- Dockerfile is correctly structured
- Shell scripts have valid syntax

No README edits required based on structural checks.

## Status: Structural verification done in CI-like environment

All AC9 requirements are **structurally satisfied** in the current codebase. However, **operator must complete all pending items on a real fresh host** (or fresh Docker container) before declaring AC9 fully verified.

**Done when:** An operator walks through the pending checklist above on a real fresh host, documents results in a new section of this file, and confirms all boxes are ✓.
