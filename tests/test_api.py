"""Tests for the FastAPI HTTP layer (src/api.py).

Strategy
--------
- Use ``TestClient`` (synchronous) to avoid nested event-loop issues.
- Override the ``get_repo`` FastAPI dependency with a ``MagicMock`` repo so
  tests never touch a real database.
- Patch ``describe_item_async`` and ``embed_async`` with ``AsyncMock`` so no
  network calls are made; every test is fully offline.
- Test the happy path for each endpoint AND the error paths (400, 404, 413).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.schemas import ItemDescription
from src.api import app, get_repo
from src.models import Item, MatchRecord

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000"
    "00907753de0000000c4944415408d76360000000000004000146a13a"
    "020000000049454e44ae426082"
)

_FAKE_DESC = ItemDescription(
    object_class="umbrella",
    colors=["black"],
    brand="Fulton",
    confidence=0.85,
)

_FAKE_VEC = np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _saved_item(id: int = 1, status: str = "lost") -> Item:
    return Item(
        id=id,
        status=status,
        filename="abc123.png",
        original_name="photo.png",
        user_text="lost umbrella",
        vlm_description=_FAKE_DESC.model_dump(),
        embedding=_FAKE_VEC.tolist(),
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_repo():
    """A mock repository with sensible defaults for all async methods."""
    repo = MagicMock()
    repo.save_image = MagicMock(return_value="abc123.png")
    repo.insert_item = AsyncMock(return_value=_saved_item())
    repo.get_item = AsyncMock(return_value=None)
    repo.list_items = AsyncMock(return_value=[])
    repo.get_embeddings_by_status = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def client(mock_repo):
    """TestClient with get_repo dependency overridden by the mock repo."""
    app.dependency_overrides[get_repo] = lambda: mock_repo
    yield TestClient(app)
    app.dependency_overrides.clear()


_PATCH_DESCRIBE = patch("src.api.describe_item_async", new_callable=AsyncMock)
_PATCH_EMBED = patch("src.api.embed_async", new_callable=AsyncMock)


def test_post_lost_returns_201(client, mock_repo):
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        response = client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    assert response.status_code == 201


def test_post_lost_response_contains_id(client, mock_repo):
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        response = client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    data = response.json()
    assert data["id"] == 1
    assert data["status"] == "lost"


def test_post_lost_saves_image(client, mock_repo):
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    mock_repo.save_image.assert_called_once()


def test_post_lost_calls_vlm_and_embed(client, mock_repo):
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    md.assert_awaited_once()
    me.assert_awaited_once()


def test_post_lost_rejects_non_image(client):
    response = client.post(
        "/items/lost",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 400


def test_post_lost_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr("src.api.settings.MAX_IMAGE_BYTES", 10)
    response = client.post(
        "/items/lost",
        files={"file": ("photo.png", _TINY_PNG, "image/png")},
    )
    assert response.status_code == 413


def test_post_lost_with_user_text(client, mock_repo):
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
            data={"user_text": "navy backpack left on bus"},
        )
    call_kwargs = md.call_args
    assert "navy backpack left on bus" in str(call_kwargs)


def test_post_found_returns_201(client, mock_repo):
    mock_repo.insert_item = AsyncMock(return_value=_saved_item(id=2, status="found"))
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        response = client.post(
            "/items/found",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    assert response.status_code == 201
    assert response.json()["status"] == "found"


def test_get_matches_returns_200(client, mock_repo):
    existing = _saved_item(id=1, status="lost")
    found_candidate = _saved_item(id=2, status="found")

    mock_repo.get_item = AsyncMock(side_effect=lambda iid: existing if iid == 1 else found_candidate)
    mock_repo.get_embeddings_by_status = AsyncMock(
        return_value=[(2, np.array([0.1, 0.2, 0.3], dtype=np.float32))]
    )

    with patch("src.api.find_matches", new_callable=AsyncMock) as mf:
        mf.return_value = [MatchRecord(candidate_id=2, score=0.98, reason="same object")]
        response = client.get("/items/1/matches?k=3")

    assert response.status_code == 200
    data = response.json()
    assert data["query_id"] == 1
    assert data["count"] == 1
    assert data["matches"][0]["candidate_id"] == 2


def test_get_matches_returns_404_for_missing_item(client, mock_repo):
    mock_repo.get_item = AsyncMock(return_value=None)
    response = client.get("/items/999/matches")
    assert response.status_code == 404


def test_get_matches_returns_422_for_item_without_embedding(client, mock_repo):
    item_no_emb = _saved_item()
    item_no_emb = item_no_emb.model_copy(update={"embedding": None})
    mock_repo.get_item = AsyncMock(return_value=item_no_emb)
    response = client.get("/items/1/matches")
    assert response.status_code == 422


def test_get_matches_k_parameter_respected(client, mock_repo):
    existing = _saved_item(id=1, status="lost")
    mock_repo.get_item = AsyncMock(return_value=existing)
    mock_repo.get_embeddings_by_status = AsyncMock(return_value=[])

    with patch("src.api.find_matches", new_callable=AsyncMock) as mf:
        mf.return_value = []
        client.get("/items/1/matches?k=5")
        _, kwargs = mf.call_args
        assert kwargs.get("k") == 5


def test_list_items_returns_200(client, mock_repo):
    mock_repo.list_items = AsyncMock(return_value=[_saved_item(1), _saved_item(2)])
    response = client.get("/items")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2


def test_list_items_filter_lost(client, mock_repo):
    mock_repo.list_items = AsyncMock(return_value=[_saved_item(1, "lost")])
    response = client.get("/items?status=lost")
    assert response.status_code == 200
    mock_repo.list_items.assert_awaited_once_with("lost")


def test_list_items_empty(client, mock_repo):
    mock_repo.list_items = AsyncMock(return_value=[])
    response = client.get("/items")
    assert response.json()["count"] == 0


def test_list_items_rejects_invalid_status(client):
    response = client.get("/items?status=unknown")
    assert response.status_code == 422


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_db_status(client, mock_repo):
    mock_repo.list_items = AsyncMock(return_value=[])
    response = client.get("/health")
    assert "db" in response.json()


def _make_insert_side_effect():
    """Return a side_effect that emulates a real DB by assigning id/created_at."""
    counter = {"n": 0}

    def _assign(item: Item) -> Item:
        counter["n"] += 1
        return item.model_copy(update={"id": counter["n"], "created_at": datetime.now(UTC)})

    return _assign


def test_post_batch_lost_returns_201(client, mock_repo):
    mock_repo.insert_item = AsyncMock(side_effect=_make_insert_side_effect())
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        response = client.post(
            "/items/batch-lost",
            files=[
                ("files", ("lost1.png", _TINY_PNG, "image/png")),
                ("files", ("lost2.png", _TINY_PNG, "image/png")),
            ],
        )
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["original_name"] == "lost1.png"
    assert data[1]["original_name"] == "lost2.png"


def test_post_batch_found_returns_201(client, mock_repo):
    mock_repo.insert_item = AsyncMock(side_effect=_make_insert_side_effect())
    with _PATCH_DESCRIBE as md, _PATCH_EMBED as me:
        md.return_value = _FAKE_DESC
        me.return_value = _FAKE_VEC
        response = client.post(
            "/items/batch-found",
            files=[
                ("files", ("found1.png", _TINY_PNG, "image/png")),
                ("files", ("found2.png", _TINY_PNG, "image/png")),
            ],
        )
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["original_name"] == "found1.png"
    assert data[1]["original_name"] == "found2.png"

