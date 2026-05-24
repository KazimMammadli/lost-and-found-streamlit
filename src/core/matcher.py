"""Business logic for matching lost items against found items.

The similarity arithmetic itself lives in :func:`ai.similarity.top_k`;
this module is responsible for two higher-level decisions:

* Choosing the candidate pool — a *lost* item is matched against the
  pool of *found* items, and vice versa. Pools are fetched from the
  repository so results always reflect the current database state.
* Producing a short, human-readable ``reason`` for each match by
  comparing the structured VLM descriptions of the two items.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ai.similarity import top_k
from src.models import Item, MatchRecord

if TYPE_CHECKING:
    from src.storage.base import AbstractRepository

log = logging.getLogger(__name__)


async def find_matches(
    item: Item,
    repo: AbstractRepository,
    k: int = 3,
) -> list[MatchRecord]:
    """Return the top-k most similar items from the opposite pool.

    Args:
        item: Query item. Must have a non-``None`` ``embedding``.
        repo: Active repository instance used to load candidate embeddings.
        k: Maximum number of matches to return.

    Returns:
        :class:`MatchRecord` instances sorted by descending cosine
        similarity. Each record carries a human-readable ``reason``
        derived from the VLM descriptions where available, or a
        score-only message otherwise.

    Raises:
        ValueError: ``item.embedding`` is ``None``, which usually means
            the item was not registered through the full AI pipeline.
    """
    if item.embedding is None:
        raise ValueError(
            f"Item {item.id} has no embedding. "
            "Ensure it was registered through the full AI pipeline."
        )

    opposite_status = "found" if item.status == "lost" else "lost"
    candidates = await repo.get_embeddings_by_status(opposite_status)

    if not candidates:
        log.info(
            "matcher.no_candidates",
            extra={"query_id": item.id, "looking_for": opposite_status},
        )
        return []

    candidate_ids = [cid for cid, _ in candidates]
    candidate_vecs = [vec for _, vec in candidates]

    query_vec = np.array(item.embedding, dtype=np.float32)
    raw_results = top_k(query_vec, candidate_vecs, k=k)

    records: list[MatchRecord] = []
    for r in raw_results:
        real_id = candidate_ids[r.candidate_id]
        clamped_score = max(0.0, min(1.0, float(r.score)))
        reason = await _build_reason(item, real_id, clamped_score, repo)
        records.append(
            MatchRecord(candidate_id=real_id, score=clamped_score, reason=reason)
        )

    log.info(
        "matcher.results",
        extra={
            "query_id": item.id,
            "k_requested": k,
            "k_returned": len(records),
            "top_score": records[0].score if records else None,
        },
    )
    return records


async def _build_reason(
    query_item: Item,
    candidate_id: int,
    score: float,
    repo: AbstractRepository,
) -> str:
    """Compose a short natural-language explanation of why two items match.

    Highlights shared object class, overlapping colours, and matching
    brand by inspecting the two ``vlm_description`` dictionaries.
    Falls back to a score-only message when descriptions are missing or
    no attributes overlap.

    Args:
        query_item: The item whose matches are being computed.
        candidate_id: Database ID of the candidate match to describe.
        score: Cosine similarity score already computed for the pair.
        repo: Repository used to load the candidate's full record.

    Returns:
        A short explanation suitable for direct display to a user.
    """
    candidate = await repo.get_item(candidate_id)

    if (
        query_item.vlm_description is None
        or candidate is None
        or candidate.vlm_description is None
    ):
        return f"Similarity score: {score:.2f}"

    q_desc = query_item.vlm_description
    c_desc = candidate.vlm_description
    parts: list[str] = []

    q_class = q_desc.get("object_class", "").lower()
    c_class = c_desc.get("object_class", "").lower()
    if q_class and c_class and q_class == c_class:
        parts.append(f"same object class ({q_class})")

    q_colors = {c.lower() for c in q_desc.get("colors", [])}
    c_colors = {c.lower() for c in c_desc.get("colors", [])}
    shared_colors = q_colors & c_colors
    if shared_colors:
        parts.append(f"shared colors: {', '.join(sorted(shared_colors))}")

    q_brand = (q_desc.get("brand") or "").lower()
    c_brand = (c_desc.get("brand") or "").lower()
    if q_brand and c_brand and q_brand == c_brand:
        parts.append(f"same brand ({q_brand})")

    if not parts:
        return f"Similarity score: {score:.2f}"

    return "; ".join(parts) + f" (score: {score:.2f})"
