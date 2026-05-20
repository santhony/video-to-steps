---
status: review_recommended
artifact_path: /Users/santhony/Documents/dev_claude/video-to-steps/PLAN.md
artifact_type: design
critics_run: ["qwen"]
critics_failed: ["deepseek"]
generated_at: 2026-05-20T22:40:30Z
total_cost_usd: 0.061884
---

# Critique: PLAN.md

**Verdict:** review_recommended
**Critics:** deepseek (failed), qwen (ok)
**Total cost:** $0.061884

## Convergent findings

_None._

## Divergent findings

_None._

## Unique findings (single-critic pass)

### hidden_assumptions
- Qwen: severity=concern, confidence=high
  - finding: The design assumes that vision-LLM-generated captions will be sufficiently accurate and consistent for reliable frame-step matching, but this may vary significantly across models and video types, affecting the core functionality in cloud mode.
  - evidence: > vision-LLM captioning ("describe what this frame shows") followed by a text embedding works through any chat-capable vision model and any text-embedding provider, which between them have broad coverage.

### failure_modes
- Qwen: severity=concern, confidence=high
  - finding: In Mode C, the design does not implement rate limiting or cost controls for vision-LLM captioning, risking API overuse and high costs during processing of long videos.
  - evidence: > Vision-LLM rate limits in Mode C. A 30-minute video at 1 fps is 1800 captioning calls. Use `asyncio.gather` with a sane semaphore (~16 concurrent) and retries with exponential backoff.

### alternatives
- Qwen: severity=note, confidence=medium
  - finding: The design does not consider using pre-extracted video embeddings from platforms like YouTube or third-party services, which could reduce computation and cost in some deployment modes.
  - evidence: > The app is designed to run in **three deployment modes** with the same code: fully local (MLX + local LLM server), hybrid (local CLIP, cloud LLM), and fully cloud-backed (vision-LLM captions + cloud LLM).

