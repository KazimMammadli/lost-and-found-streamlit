"""Tests for src/core/matcher.py.

The matcher is pure business logic — it queries the repository for embeddings
and delegates the math to ``ai.similarity.top_k``.  We inject a fake
repository and controlled embeddings so tests are deterministic and offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.core.matcher import find_matches
from src.models import Item, MatchRecord


def _item(
    id: int,
    status: str,
    embedding: list[float] | None = None,
    vlm_description: dict | None = None,
) -> Item:
    return Item(
        id=id,
        status=status,
        filename=f"img{id}.png",
        original_name=f"photo{id}.png",
        vlm_description=vlm_description
        or {"object_class": "backpack", "colors": ["navy"], "confidence": 0.9},
        embedding=embedding,
        created_at=datetime.now(UTC),
    )


def _unit(vec: list[float]) -> np.ndarray:
    """Return a unit-normalized vector."""
    a = np.array(vec, dtype=np.float32)
    return a / np.linalg.norm(a)


def _mock_repo(
    found_items: list[Item] | None = None,
) -> AsyncMock:
    """Return a mock repository pre-loaded with found items."""
    repo = AsyncMock()

    embeddings: list[tuple[int, np.ndarray]] = []
    for item in (found_items or []):
        if item.embedding is not None:
            embeddings.append((item.id, np.array(item.embedding, dtype=np.float32)))

    repo.get_embeddings_by_status = AsyncMock(return_value=embeddings)

    async def _get_item(item_id: int) -> Item | None:
        all_items = found_items or []
        return next((i for i in all_items if i.id == item_id), None)

    repo.get_item = _get_item
    return repo


async def test_find_matches_raises_if_no_embedding():
    repo = _mock_repo()
    item = _item(1, "lost", embedding=None)
    with pytest.raises(ValueError, match="embedding"):
        await find_matches(item, repo)


async def test_find_matches_returns_empty_when_pool_empty():
    repo = _mock_repo(found_items=[])
    item = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist())
    results = await find_matches(item, repo)
    assert results == []


async def test_find_matches_lost_searches_found_pool():
    """A 'lost' item should be matched against 'found' items."""
    repo = _mock_repo()
    item = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist())
    await find_matches(item, repo)
    repo.get_embeddings_by_status.assert_awaited_once_with("found")


async def test_find_matches_found_searches_lost_pool():
    """A 'found' item should be matched against 'lost' items."""
    repo = _mock_repo()
    item = _item(1, "found", embedding=_unit([1.0, 0.0]).tolist())
    await find_matches(item, repo)
    repo.get_embeddings_by_status.assert_awaited_once_with("lost")


async def test_find_matches_returns_match_records():
    found = [_item(10, "found", embedding=_unit([1.0, 0.0]).tolist())]
    repo = _mock_repo(found)
    lost = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist())
    results = await find_matches(lost, repo, k=1)
    assert len(results) == 1
    assert isinstance(results[0], MatchRecord)
    assert results[0].candidate_id == 10


async def test_find_matches_sorted_descending():
    """Results must be ranked highest score first."""
    found = [
        _item(10, "found", embedding=_unit([0.0, 1.0]).tolist()),
        _item(11, "found", embedding=_unit([1.0, 0.0]).tolist()),
        _item(12, "found", embedding=_unit([0.7, 0.7]).tolist()),
    ]
    repo = _mock_repo(found)
    query = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist())
    results = await find_matches(query, repo, k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_find_matches_top_candidate_is_most_similar():
    found = [
        _item(10, "found", embedding=_unit([1.0, 0.0]).tolist()),
        _item(11, "found", embedding=_unit([0.0, 1.0]).tolist()),
    ]
    repo = _mock_repo(found)
    query = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist())
    results = await find_matches(query, repo, k=2)
    assert results[0].candidate_id == 10


async def test_find_matches_respects_k_limit():
    found = [_item(i, "found", embedding=_unit([float(i), 0.0]).tolist()) for i in range(1, 6)]
    repo = _mock_repo(found)
    query = _item(99, "lost", embedding=_unit([1.0, 0.0]).tolist())
    results = await find_matches(query, repo, k=2)
    assert len(results) <= 2


async def test_reason_includes_shared_color():
    desc_query = {"object_class": "backpack", "colors": ["navy", "black"], "confidence": 0.9}
    desc_cand = {"object_class": "backpack", "colors": ["black", "red"], "confidence": 0.8}
    found = [_item(10, "found", embedding=_unit([1.0, 0.0]).tolist(), vlm_description=desc_cand)]
    repo = _mock_repo(found)
    query = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist(), vlm_description=desc_query)
    results = await find_matches(query, repo, k=1)
    assert "black" in results[0].reason


async def test_reason_includes_same_object_class():
    desc = {"object_class": "umbrella", "colors": ["red"], "confidence": 0.9}
    found = [_item(10, "found", embedding=_unit([1.0, 0.0]).tolist(), vlm_description=desc)]
    repo = _mock_repo(found)
    query = _item(1, "lost", embedding=_unit([1.0, 0.0]).tolist(), vlm_description=desc)
    results = await find_matches(query, repo, k=1)
    assert "umbrella" in results[0].reason


async def test_reason_fallback_when_no_description():
    """Items without vlm_description get a score-only reason."""
    item_no_desc = Item(
        id=10, status="found",
        filename="x.png", original_name="x.png",
        embedding=_unit([1.0, 0.0]).tolist(),
        created_at=datetime.now(UTC),
    )
    repo = AsyncMock()
    repo.get_embeddings_by_status = AsyncMock(
        return_value=[(10, np.array(item_no_desc.embedding, dtype=np.float32))]
    )
    repo.get_item = AsyncMock(return_value=item_no_desc)

    query = Item(
        id=1, status="lost", filename="q.png", original_name="q.png",
        embedding=_unit([1.0, 0.0]).tolist(),
        created_at=datetime.now(UTC),
    )
    results = await find_matches(query, repo, k=1)
    assert "score" in results[0].reason.lower() or results[0].reason != ""
