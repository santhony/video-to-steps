"""End-to-end orchestrator: URL in, steps.json out, meta.json updated.

The orchestrator's job is composition. It does not implement any pipeline
logic itself — every stage delegates to a module from earlier phases.

Manifest lifecycle:
- 'queued' → set on creation by the server (not here)
- 'running' → set immediately at run_job start
- 'done' → set on successful completion
- 'error' → set in the outer try/except OR by the captionless-video early-return

Pattern: Imperative Shell
This module coordinates I/O operations (stage invocations) and manifest
persistence while pure logic lives in earlier phases. The orchestrator is
responsible for error handling, state transitions, and cost accumulation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Settings
from pipeline.caption_winners import caption_winners
from pipeline.captions import dedupe_rolling, parse_vtt
from pipeline.download import download_video_and_captions
from pipeline.frames import FixedFpsExtractor
from pipeline.llm_outline import llm_outline
from pipeline.llm_refine import llm_refine
from pipeline.match import match
from pipeline.storage import ensure_job_dir, write_json_atomic
from pipeline.types import CostBreakdown, Manifest
from pricing import compute_chat_cost, compute_embed_cost, compute_vision_cost
from providers.embed import build_embedder
from providers.llm import build_llm
from providers.vision import build_vision

log = logging.getLogger(__name__)


def _update(manifest: Manifest, jobs_root: Path, **fields) -> None:
    """Update manifest fields and persist to meta.json atomically."""
    for k, v in fields.items():
        setattr(manifest, k, v)
    write_json_atomic(jobs_root / manifest.job_id / "meta.json", manifest)


def _config_snapshot(settings: Settings) -> dict:
    """Capture the runtime configuration for this job."""
    return {
        "embed_backend": settings.embed_backend,
        "llm_model": settings.llm_model,
        "vision_model": settings.vision_model,
        "jina_model": settings.jina_model,
        "refine_max_in_flight": settings.refine_max_in_flight,
        "caption_max_in_flight": settings.caption_max_in_flight,
        "whisper_fallback": settings.whisper_fallback,
    }


def _mode_label(settings: Settings) -> str:
    """Classify the runtime mode based on embedder + LLM backend."""
    if settings.embed_backend == "mlx_clip":
        return "hybrid" if not settings.llm_base_url.startswith("http://127.") else "local"
    return "cloud"


async def run_job(job_id: str, url: str, settings: Settings, jobs_root: Path) -> None:
    """Runs the full pipeline for one job. Writes meta.json + steps.json."""
    job_dir = ensure_job_dir(jobs_root, job_id)
    manifest = Manifest(
        job_id=job_id,
        url=url,
        status="running",
        progress="starting",
        mode=_mode_label(settings),
        config_snapshot=_config_snapshot(settings),
    )
    _update(manifest, jobs_root)

    try:
        # ── Stage 1: download ──────────────────────────────────────────────
        _update(manifest, jobs_root, progress="downloading video")
        video, vtt = download_video_and_captions(url, job_dir)

        if vtt is None:
            if not settings.whisper_fallback:
                _update(
                    manifest, jobs_root,
                    status="error",
                    progress="",
                    error="This video has no captions. Whisper fallback is on the v2 roadmap.",
                )
                return
            raise RuntimeError("Whisper fallback enabled but not implemented in v1.")

        # ── Stage 2: parse + dedupe captions ───────────────────────────────
        _update(manifest, jobs_root, progress="parsing captions")
        cues = dedupe_rolling(parse_vtt(vtt))

        # ── Stage 3: extract + dedupe frames ───────────────────────────────
        _update(manifest, jobs_root, progress="extracting frames")
        extractor = FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6)
        frames = extractor.extract(video, job_dir / "frames")

        # ── Stage 4: embed every frame ─────────────────────────────────────
        _update(manifest, jobs_root, progress="embedding frames")
        embedder = build_embedder(settings)
        frame_paths = [f.path for f in frames]
        frame_res = await embedder.embed_images(frame_paths)
        np.save(job_dir / "frame_embeddings.npy", frame_res.vectors)

        # ── Stage 5: outline ──────────────────────────────────────────────
        _update(manifest, jobs_root, progress="outlining steps")
        llm = build_llm(settings)
        outlines, outline_usage = await llm_outline(cues, llm)
        write_json_atomic(job_dir / "outline.json", [asdict(o) for o in outlines])

        # ── Stage 6: embed step briefs ────────────────────────────────────
        _update(manifest, jobs_root, progress="embedding step briefs")
        step_res = await embedder.embed_texts([o.brief for o in outlines])

        # ── Stage 7: match frames to steps ─────────────────────────────────
        _update(manifest, jobs_root, progress="matching frames to steps")
        winners = match(outlines, frames, frame_res.vectors, step_res.vectors)
        winners_by_step = {o.index: ws for o, ws in zip(outlines, winners)}

        # ── Stage 8: caption winning frames ────────────────────────────────
        _update(manifest, jobs_root, progress="captioning representative frames")
        captioner = build_vision(settings)
        captions, caption_usage = await caption_winners(
            winners_by_step, job_dir, captioner,
            max_in_flight=settings.caption_max_in_flight,
        )

        # ── Stage 9: refine each step ──────────────────────────────────────
        _update(manifest, jobs_root, progress="refining step text")
        steps, refine_usage = await llm_refine(
            outlines=outlines,
            cues=cues,
            winners_by_step=winners_by_step,
            captions=captions,
            llm=llm,
            max_in_flight=settings.refine_max_in_flight,
        )

        # ── Stage 10: persist + accumulate cost ────────────────────────────
        write_json_atomic(job_dir / "steps.json", [asdict(s) for s in steps])

        chat_cost = (
            compute_chat_cost(settings.llm_model, outline_usage.prompt_tokens, outline_usage.completion_tokens)
            + compute_chat_cost(settings.llm_model, refine_usage.prompt_tokens, refine_usage.completion_tokens)
        )
        vision_cost = compute_vision_cost(
            settings.vision_model, caption_usage.prompt_tokens, caption_usage.completion_tokens
        )
        embed_cost = (
            compute_embed_cost(embedder.name, frame_res.billable_tokens)
            + compute_embed_cost(embedder.name, step_res.billable_tokens)
        )
        cost = CostBreakdown(
            chat_usd=chat_cost,
            vision_usd=vision_cost,
            embed_usd=embed_cost,
            total_usd=chat_cost + vision_cost + embed_cost,
        )

        _update(manifest, jobs_root, status="done", progress="done", cost=cost, error="")

    except Exception as exc:  # noqa: BLE001 — orchestrator policy: surface, don't paper over.
        log.exception("run_job %s failed", job_id)
        _update(
            manifest, jobs_root,
            status="error",
            progress="",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        # Close any provider clients that opened HTTP sessions. Mode A's
        # MlxClipEmbedder has no client to close.
        for closer in (
            getattr(locals().get("embedder"), "aclose", None),
            getattr(locals().get("llm"), "aclose", None),
            getattr(locals().get("captioner"), "aclose", None),
        ):
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass
