"""Smoke-test the configured LLM endpoint.

Usage: python scripts/smoke_llm.py

Reads .env via config.Settings; calls LLMClient once; prints token counts
and a price-table-derived cost estimate. Non-zero exit on failure.
"""

from __future__ import annotations

import asyncio
import sys

from config import get_settings
from pricing import compute_chat_cost
from providers.llm import build_llm


async def main() -> int:
    settings = get_settings()
    llm = build_llm(settings)
    try:
        result = await llm.chat([{"role": "user", "content": "Say 'pong' and nothing else."}])
    except Exception as exc:  # noqa: BLE001 — operator-facing one-liner is the point
        print(f"smoke_llm FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await llm.aclose()

    cost = compute_chat_cost(settings.llm_model, result.prompt_tokens, result.completion_tokens)
    print(f"text:              {result.text.strip()!r}")
    print(f"prompt_tokens:     {result.prompt_tokens}")
    print(f"completion_tokens: {result.completion_tokens}")
    print(f"est_cost_usd:      {cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
