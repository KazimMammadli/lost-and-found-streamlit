"""Tests for the storage layer (SQLiteRepository).

All tests use an in-memory SQLite database so they run offline and leave no
files on disk.  The same ``AbstractRepository`` interface is tested here —
the PostgreSQL implementation satisfies the same contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import Item
from src.storage.sqlite_repo import SQLiteRepository

# ------------------------------------------------------------------ #
# Fixtures                                                              #
# ------------------------------------------------------------------ #


@pytest.fixture
async def repo(tmp_path) -> SQLiteRepository:
    """In-memory SQLite repository, initialised and torn down per test."""
    r = SQLiteRepository(":memory:")
    await r.initialize()
    yield r
    await r.close()


def _make_item(status="lost", with_embedding=True) -> Item:
    """Factory for a minimal valid Item domain model."""
    embedding = [0.1, 0.2, 0.3] if with_embedding else None
    return Item(
        status=status,
        filename="abc123.png",
        original_name="photo.png",
        user_text="navy backpack",
        vlm_description={"object_class": "backpack", "colors": ["navy"], "confidence": 0.9},
        embedding=embedding,
    )


# ------------------------------------------------------------------ #
# insert_item                                                           #
# ------------------------------------------------------------------ #


async def test_insert_item_assigns_id(repo):
    item = _make_item()
    saved = await repo.insert_item(item)
    assert saved.id is not None
    assert saved.id >= 1


async def test_insert_item_sets_created_at(repo):
    saved = await repo.insert_item(_make_item())
    assert saved.created_at is not None


async def test_insert_two_items_get_distinct_ids(repo):
    a = await repo.insert_item(_make_item("lost"))
    b = await repo.insert_item(_make_item("found"))
    assert a.id != b.id


async def test_insert_preserves_status(repo):
    saved = await repo.insert_item(_make_item("found"))
    assert saved.status == "found"


async def test_insert_preserves_vlm_description(repo):
    saved = await repo.insert_item(_make_item())
    assert saved.vlm_description is not None
    assert saved.vlm_description["object_class"] == "backpack"


async def test_insert_preserves_embedding(repo):
    saved = await repo.insert_item(_make_item())
    assert saved.embedding is not None
    assert len(saved.embedding) == 3


async def test_insert_item_without_embedding(repo):
    saved = await repo.insert_item(_make_item(with_embedding=False))
    assert saved.embedding is None


# ------------------------------------------------------------------ #
# get_item                                                              #
# ------------------------------------------------------------------ #


async def test_get_item_returns_correct_item(repo):
    saved = await repo.insert_item(_make_item())
    fetched = await repo.get_item(saved.id)
    assert fetched is not None
    assert fetched.id == saved.id
    assert fetched.status == saved.status


async def test_get_item_returns_none_for_missing_id(repo):
    result = await repo.get_item(99999)
    assert result is None


async def test_get_item_restores_embedding(repo):
    saved = await repo.insert_item(_make_item())
    fetched = await repo.get_item(saved.id)
    assert fetched.embedding is not None
    assert np.allclose(fetched.embedding, [0.1, 0.2, 0.3], atol=1e-5)


async def test_get_item_restores_vlm_description(repo):
    saved = await repo.insert_item(_make_item())
    fetched = await repo.get_item(saved.id)
    assert fetched.vlm_description["colors"] == ["navy"]


# ------------------------------------------------------------------ #
# list_items                                                            #
# ------------------------------------------------------------------ #


async def test_list_items_returns_all(repo):
    await repo.insert_item(_make_item("lost"))
    await repo.insert_item(_make_item("found"))
    items = await repo.list_items()
    assert len(items) == 2


async def test_list_items_filter_lost(repo):
    await repo.insert_item(_make_item("lost"))
    await repo.insert_item(_make_item("found"))
    items = await repo.list_items("lost")
    assert all(i.status == "lost" for i in items)
    assert len(items) == 1


async def test_list_items_filter_found(repo):
    await repo.insert_item(_make_item("found"))
    items = await repo.list_items("found")
    assert len(items) == 1


async def test_list_items_all_alias(repo):
    await repo.insert_item(_make_item("lost"))
    await repo.insert_item(_make_item("found"))
    items = await repo.list_items("all")
    assert len(items) == 2


async def test_list_items_empty_db(repo):
    items = await repo.list_items()
    assert items == []


# ------------------------------------------------------------------ #
# get_embeddings_by_status                                              #
# ------------------------------------------------------------------ #


async def test_get_embeddings_returns_correct_status(repo):
    lost = await repo.insert_item(_make_item("lost"))
    await repo.insert_item(_make_item("found"))
    embeddings = await repo.get_embeddings_by_status("lost")
    ids = [e[0] for e in embeddings]
    assert lost.id in ids
    assert len(embeddings) == 1


async def test_get_embeddings_skips_items_without_embedding(repo):
    await repo.insert_item(_make_item(with_embedding=False))
    embeddings = await repo.get_embeddings_by_status("lost")
    assert embeddings == []


async def test_get_embeddings_returns_numpy_arrays(repo):
    await repo.insert_item(_make_item("found"))
    embeddings = await repo.get_embeddings_by_status("found")
    assert len(embeddings) == 1
    _, vec = embeddings[0]
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32


# ------------------------------------------------------------------ #
# save_image                                                            #
# ------------------------------------------------------------------ #


def test_save_image_returns_uuid_filename(repo, tmp_path, monkeypatch):
    monkeypatch.setattr("src.storage.base.settings", type("S", (), {"IMAGES_DIR": str(tmp_path)})())
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 80
    name = repo.save_image(data, "my photo.png")
    assert name.endswith(".png")
    assert (tmp_path / name).exists()


def test_save_image_rejects_path_traversal(repo, tmp_path, monkeypatch):
    monkeypatch.setattr("src.storage.base.settings", type("S", (), {"IMAGES_DIR": str(tmp_path)})())
    # Even a traversal name should produce a safe UUID-based path
    data = b"\x89PNG"
    name = repo.save_image(data, "../../etc/passwd")
    # The resulting name must not contain '..'
    assert ".." not in name


async def test_repo_raises_if_not_initialized():
    repo = SQLiteRepository(":memory:")
    with pytest.raises(RuntimeError, match="initialised"):
        await repo.get_item(1)
