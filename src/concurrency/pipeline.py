"""Async concurrency primitives for batch operations.

The central building block is :func:`run_with_semaphore`, which runs a
collection of async callables in parallel while capping in-flight work
at ``settings.SEMAPHORE_LIMIT``. The cap prevents flooding the AI
provider with more requests than it can absorb and avoids HTTP 429
errors under load.

A semaphore — rather than a thread pool — is used here because the AI
calls themselves are already dispatched to a thread pool inside
:func:`describe_item_async` via :func:`asyncio.to_thread`. The
semaphore controls how many of those thread dispatches are in-flight at
once, which is what the provider's rate limit cares about.

:func:`register_batch` is the high-level entry point used by the batch
API endpoints (``/items/batch-lost``, ``/items/batch-found``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Literal

from src.config import settings

log = logging.getLogger(__name__)


async def run_with_semaphore(
    coro_factories: list[Callable[[], Coroutine[Any, Any, Any]]],
) -> list[Any]:
    """Run a list of async callables concurrently with bounded parallelism.

    Args:
        coro_factories: List of zero-argument callables, each returning a
            coroutine. Using factories rather than bare coroutines lets
            the caller decide when each coroutine is *created*, which
            matters for resource-heavy setup (file reads, image decoding).

    Returns:
        Results in the same order as ``coro_factories``. Exceptions are
        returned as values rather than raised, so callers can inspect
        partial failures and decide how to handle them.
    """
    sem = asyncio.Semaphore(settings.SEMAPHORE_LIMIT)

    async def _guarded(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        """Execute ``factory()`` while holding the semaphore."""
        async with sem:
            return await factory()

    tasks = [asyncio.create_task(_guarded(f)) for f in coro_factories]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    log.debug(
        "pipeline.run_with_semaphore.done",
        extra={
            "total": len(results),
            "errors": sum(1 for r in results if isinstance(r, Exception)),
        },
    )
    return list(results)


async def register_batch(
    image_files: list[tuple[bytes, str]],
    status: Literal["lost", "found"],
    repo: Any,
    user_texts: list[str | None] | None = None,
) -> list[Any]:
    """Register multiple images concurrently with bounded parallelism.

    This is the batch counterpart to the per-item registration pipeline
    used by the single-image HTTP endpoints. Each image is saved to disk,
    described by the VLM, embedded, and inserted into the repository
    independently of the others.

    Args:
        image_files: ``(image_bytes, original_filename)`` pairs.
        status: ``"lost"`` or ``"found"`` — applied to every item.
        repo: Active repository used for image storage and inserts.
        user_texts: Optional per-image free-text descriptions. Shorter
            lists are padded with ``None``.

    Returns:
        One entry per input image, in input order. Successful entries are
        the persisted :class:`Item`; failures appear as :class:`Exception`
        instances so callers can report partial success without halting.
    """
    from src.models import Item
    from src.services.ai_service import describe_item_async, embed_async

    texts = user_texts or []

    async def _one(data: bytes, name: str, text: str | None) -> Item:
        """Register a single image and return the persisted :class:`Item`."""
        filename = repo.save_image(data, name)
        image_path = str(Path(settings.IMAGES_DIR) / filename)

        description = await describe_item_async(image_path, text or "")
        embedding_vec = await embed_async(description.to_search_text())

        item = Item(
            status=status,
            filename=filename,
            original_name=name,
            user_text=text,
            vlm_description=description.to_dict(),
            embedding=embedding_vec.tolist(),
        )
        return await repo.insert_item(item)

    factories: list[Callable[[], Coroutine[Any, Any, Any]]] = []
    for i, (data, name) in enumerate(image_files):
        t = texts[i] if i < len(texts) else None
        factories.append(
            (lambda d=data, n=name, txt=t: _one(d, n, txt))  # type: ignore[misc]
        )

    return await run_with_semaphore(factories)
