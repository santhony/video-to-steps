"""Per-model pricing for cost reporting.

Edit `PRICES` to add new models. Missing models record zeros and log a
warning at startup; the pipeline still completes.

All prices are USD per 1,000,000 tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    prompt_per_million: float
    completion_per_million: float
    embed_per_million: float = 0.0


# Last reviewed: 2026-05. Spot-check against provider pricing pages before
# trusting absolute numbers for budgeting; the goal here is rough cost
# visibility per run, not finance-grade accounting.
PRICES: dict[str, ModelPrice] = {
    # Text LLM
    "deepseek-chat":      ModelPrice(prompt_per_million=0.27, completion_per_million=1.10),
    "deepseek-v4-flash":  ModelPrice(prompt_per_million=0.07, completion_per_million=0.28),
    "deepseek-v4-pro":    ModelPrice(prompt_per_million=0.55, completion_per_million=2.19),
    "gpt-4o-mini":        ModelPrice(prompt_per_million=0.15, completion_per_million=0.60),
    "gpt-4o":             ModelPrice(prompt_per_million=2.50, completion_per_million=10.00),
    # Vision
    # (gpt-4o-mini is dual-use; same price table entry as above)
    "meta-llama/Llama-Vision-Free": ModelPrice(prompt_per_million=0.0, completion_per_million=0.0),
    # Embeddings — Jina charges per token; image tokens depend on tile count.
    "jina-embeddings-v4": ModelPrice(prompt_per_million=0.0, completion_per_million=0.0, embed_per_million=0.18),
}


# Module-scoped: warnings dedupe within a single Python process. Tests that
# care about exact warning emission should use unique model names.
_warned: set[str] = set()


def _warn_once(model_id: str) -> None:
    if model_id not in _warned:
        _warned.add(model_id)
        log.warning("pricing.py: no entry for model %r; cost will record zero.", model_id)


def _price_or_zero(model_id: str) -> ModelPrice:
    p = PRICES.get(model_id)
    if p is None:
        _warn_once(model_id)
        return ModelPrice(0.0, 0.0, 0.0)
    return p


def compute_chat_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = _price_or_zero(model_id)
    return (prompt_tokens / 1_000_000.0) * p.prompt_per_million \
         + (completion_tokens / 1_000_000.0) * p.completion_per_million


def compute_vision_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Vision is just a chat call with image parts; same pricing shape.
    return compute_chat_cost(model_id, prompt_tokens, completion_tokens)


def compute_embed_cost(model_id: str, total_tokens: int) -> float:
    p = _price_or_zero(model_id)
    return (total_tokens / 1_000_000.0) * p.embed_per_million
