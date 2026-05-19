"""Smoke-test the configured vision endpoint with a checked-in test frame.

Usage: python scripts/smoke_vision.py

Reads .env via config.Settings; calls VisionCaptioner once on a test frame;
prints caption text, token counts and a price-table-derived cost estimate.
Non-zero exit on failure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config import get_settings
from pricing import compute_vision_cost
from providers.vision import build_vision


FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "providers" / "fixtures" / "test_frame.jpg"


async def main() -> int:
    settings = get_settings()
    vis = build_vision(settings)
    try:
        result = await vis.caption(FIXTURE)
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_vision FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await vis.aclose()

    cost = compute_vision_cost(settings.vision_model, result.prompt_tokens, result.completion_tokens)
    print(f"caption:           {result.text.strip()!r}")
    print(f"prompt_tokens:     {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
