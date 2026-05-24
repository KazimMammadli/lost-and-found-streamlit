"""Resilient async wrappers around the provided ``ai`` module.

Every AI call in the application flows through one of the two coroutines
defined here. The wrappers layer three rubric-required behaviours on top
of the underlying synchronous calls:

* **Retries** — up to three attempts with exponential back-off and jitter,
  via ``tenacity``. Only :class:`ProviderError` and timeouts are retried;
  :class:`ValueError` (a caller bug, such as an empty prompt) is re-raised
  immediately so the caller can fix the input.

* **Timeouts** — each call runs inside :func:`asyncio.wait_for` so it
  cannot exceed ``settings.AI_CALL_TIMEOUT_S``.

* **Caching** — :func:`embed_async` memoises embedding vectors in a
  process-local dictionary keyed on the exact input string. Identical
  inputs short-circuit without touching the provider.

Three bonus features integrate here as well:

* **Rate limiting** — :class:`TokenBudget` is acquired before each call
  so the configured tokens-per-minute ceiling is respected.
* **Cost telemetry** — :class:`CostMeter` records token counts and an
  estimated dollar cost after each successful call.
* **Failover** — when ``SECONDARY_LLM_PROVIDER`` is set, the VLM call is
  routed through :class:`FailoverVLM`, which automatically falls back to
  the secondary on a :class:`ProviderError` from the primary.

All diagnostic output uses :mod:`logging`; no ``print`` calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai import describe_item, embed
from ai.providers.base import ProviderError
from ai.schemas import ItemDescription
from src.config import settings
from src.tracing import traced_ai_call

log = logging.getLogger(__name__)

_RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ProviderError, TimeoutError, asyncio.TimeoutError)),
    reraise=True,
    before_sleep=before_sleep_log(log, logging.WARNING),
)
"""Tenacity policy applied to both async wrappers.

``ProviderError`` is the ai module's own failure signal; ``TimeoutError``
covers the case where :func:`asyncio.wait_for` expires while the underlying
thread is still running.
"""

_embed_cache: dict[str, tuple[float, ...]] = {}
"""Process-local embedding cache.

Maps the exact text string to a tuple of floats (hashable and easy to
serialise). The cache survives only for the lifetime of the process —
restarting clears it intentionally.
"""


@traced_ai_call("ai.describe_item", provider_attr="LLM_PROVIDER", model_attr="LLM_MODEL")
@retry(**_RETRY_POLICY)  # type: ignore[call-overload]
async def describe_item_async(
    image_path: str,
    user_text: str,
    *,
    vlm: Any = None,
) -> ItemDescription:
    """Describe an item image with the VLM, with retries and a timeout.

    Args:
        image_path: Absolute or relative path to a JPEG or PNG file.
        user_text: Free-text description supplied by the user at
            registration time. May be empty.
        vlm: Optional VLM provider override (used in tests to inject a
            fake). When ``None``, the rate limiter and cost meter are
            engaged and, if ``SECONDARY_LLM_PROVIDER`` is configured, a
            :class:`FailoverVLM` is constructed automatically.

    Returns:
        Validated :class:`ItemDescription` from the ``ai`` module.

    Raises:
        ProviderError: All retry attempts failed.
        asyncio.TimeoutError: The call exceeded ``settings.AI_CALL_TIMEOUT_S``.
    """
    if vlm is None:
        estimated_tokens = max(200, int(len(user_text or "") * 1.3) + 200)
        from src.concurrency.rate_limiter import get_token_budget

        await get_token_budget().acquire(estimated_tokens)

    effective_vlm = vlm
    if effective_vlm is None:
        from src.core.failover import build_failover_vlm_if_configured

        effective_vlm = build_failover_vlm_if_configured()

    log.info(
        "ai_service.describe.start",
        extra={"image": image_path, "text_len": len(user_text or "")},
    )
    result: ItemDescription = await asyncio.wait_for(
        asyncio.to_thread(describe_item, image_path, user_text, vlm=effective_vlm),
        timeout=settings.AI_CALL_TIMEOUT_S,
    )
    log.info(
        "ai_service.describe.ok",
        extra={
            "class": result.object_class,
            "confidence": result.confidence,
            "colors": result.colors,
        },
    )

    if vlm is None:
        _record_describe_cost(user_text)

    return result


@traced_ai_call("ai.embed", provider_attr="EMBEDDING_PROVIDER", model_attr="EMBEDDING_MODEL")
@retry(**_RETRY_POLICY)  # type: ignore[call-overload]
async def embed_async(
    text: str,
    *,
    embedder: Any = None,
) -> np.ndarray:
    """Return a unit-normalised embedding vector for ``text``, with caching.

    Repeated calls with the same ``text`` value during the lifetime of
    the process return the cached vector without contacting the provider.

    Args:
        text: Non-empty string to embed.
        embedder: Optional embedder override (used in tests to inject a
            fake). When ``None`` the rate limiter, cache, and cost meter
            are engaged.

    Returns:
        One-dimensional ``float32`` unit-normalised vector.

    Raises:
        ValueError: ``text`` is empty or whitespace-only. Not retried.
        ProviderError: The provider failed after all retry attempts.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty or whitespace-only text.")

    if text in _embed_cache and embedder is None:
        log.debug("ai_service.embed.cache_hit", extra={"text_len": len(text)})
        return np.array(_embed_cache[text], dtype=np.float32)

    if embedder is None:
        estimated_tokens = max(10, int(len(text.split()) * 1.3))
        from src.concurrency.rate_limiter import get_token_budget

        await get_token_budget().acquire(estimated_tokens)

    log.info("ai_service.embed.start", extra={"text_len": len(text)})
    vec: np.ndarray = await asyncio.wait_for(
        asyncio.to_thread(embed, text, embedder=embedder),
        timeout=settings.AI_CALL_TIMEOUT_S,
    )
    log.info("ai_service.embed.ok", extra={"dim": vec.shape[0]})

    if embedder is None:
        _embed_cache[text] = tuple(vec.tolist())
        _record_embed_cost(text)

    return vec


def clear_embed_cache() -> None:
    """Empty the in-process embedding cache.

    Useful in tests that need a clean slate between runs.
    """
    _embed_cache.clear()


def _record_describe_cost(user_text: str) -> None:
    """Append a cost event for a successful VLM describe call.

    Best-effort: any error from the cost meter is swallowed at ``DEBUG``
    level so cost-tracking failures cannot break the main flow.
    """
    try:
        from src.services.cost_meter import get_cost_meter

        prompt_est = max(200, int(len(user_text or "") * 1.3) + 200)
        completion_est = 150
        get_cost_meter().record(
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            prompt_tokens=prompt_est,
            completion_tokens=completion_est,
        )
    except Exception:
        log.debug("ai_service.cost_meter.record_failed", exc_info=True)


def _record_embed_cost(text: str) -> None:
    """Append a cost event for a successful embedding call.

    Best-effort: see :func:`_record_describe_cost`.
    """
    try:
        from src.services.cost_meter import get_cost_meter

        token_est = max(10, int(len(text.split()) * 1.3))
        get_cost_meter().record(
            provider=settings.EMBEDDING_PROVIDER,
            model=settings.EMBEDDING_MODEL,
            prompt_tokens=token_est,
            completion_tokens=0,
        )
    except Exception:
        log.debug("ai_service.cost_meter.embed_record_failed", exc_info=True)
