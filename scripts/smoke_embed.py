"""Smoke-test the configured embedder.

Embeds one text and one frame (the checked-in fixture) and prints the
output dimensionality, vector dtype, L2-norm of the first row, and a
cost estimate.

Usage: python scripts/smoke_embed.py

Reads .env via config.Settings; calls the configured embedder once;
prints vector shapes, L2 norms, token counts and a price-table-derived cost
estimate. Non-zero exit on failure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

from config import get_settings
from pricing import compute_embed_cost
from providers.embed import build_embedder


FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "providers" / "fixtures" / "test_frame.jpg"


async def main() -> int:
    settings = get_settings()
    emb = build_embedder(settings)
    try:
        text_res = await emb.embed_texts(["a hand chopping an onion"])
        image_res = await emb.embed_images([FIXTURE])
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_embed FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        aclose = getattr(emb, "aclose", None)
        if aclose is not None:
            await aclose()

    total_tokens = text_res.billable_tokens + image_res.billable_tokens
    cost = compute_embed_cost(emb.name, total_tokens)
    print(f"backend:           {emb.name}")
    print(f"text dim:          {text_res.vectors.shape}")
    print(f"image dim:         {image_res.vectors.shape}")
    print(f"text L2:           {float(np.linalg.norm(text_res.vectors[0])):.6f}")
    print(f"image L2:          {float(np.linalg.norm(image_res.vectors[0])):.6f}")
    print(f"billable_tokens:   {total_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
