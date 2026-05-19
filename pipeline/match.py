"""Frame-to-step matching.

Pure function: given per-step time bounds and L2-normalized embeddings for
both frames and steps, return per-step top-k Frame winners. The candidate
window restriction respects each step's [start - pad, end + pad] span
before scoring. When that window is empty, fall back to the single frame
nearest the step midpoint.

pattern: Functional Core
"""

from __future__ import annotations

import numpy as np

from .types import Frame, StepOutline


def match(
    steps: list[StepOutline],
    frames: list[Frame],
    frame_emb: np.ndarray,         # shape (n_frames, d), L2-normalized float32
    step_emb: np.ndarray,          # shape (n_steps, d),  L2-normalized float32
    *,
    pad_sec: float = 2.0,
    top_k: int = 3,
) -> list[list[Frame]]:
    """Returns winners-per-step in the same order as `steps`.

    Cosine similarity between an L2-normalized frame vector and an
    L2-normalized step vector IS their dot product — so the scoring is a
    plain `frame_emb @ step_emb_row`.
    """
    if not steps:
        return []
    if not frames:
        return [[] for _ in steps]
    if frame_emb.shape[0] != len(frames):
        raise ValueError(f"frame_emb rows ({frame_emb.shape[0]}) != len(frames) ({len(frames)})")
    if step_emb.shape[0] != len(steps):
        raise ValueError(f"step_emb rows ({step_emb.shape[0]}) != len(steps) ({len(steps)})")

    frame_times = np.array([f.timestamp for f in frames], dtype=np.float32)
    out: list[list[Frame]] = []

    for i, step in enumerate(steps):
        lo, hi = step.start - pad_sec, step.end + pad_sec
        in_window = np.where((frame_times >= lo) & (frame_times <= hi))[0]
        if in_window.size == 0:
            # Empty window: nearest-to-midpoint fallback.
            mid = (step.start + step.end) / 2.0
            nearest = int(np.argmin(np.abs(frame_times - mid)))
            out.append([frames[nearest]])
            continue

        scores = frame_emb[in_window] @ step_emb[i]
        order = np.argsort(-scores)              # descending
        winners_local = in_window[order][:top_k]
        out.append([frames[int(j)] for j in winners_local])

    return out
