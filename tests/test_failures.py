"""Failure-injection tests for the HTTP API layer.

These tests verify that the system degrades gracefully when:
- The AI provider fails (HTTP 502)
- A file is the wrong type (HTTP 400)
- A file is too large (HTTP 413)
- An item is not found (HTTP 404)

All tests are offline — no network calls are made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.providers.base import ProviderError
from ai.schemas import ItemDescription
from src.api import app, get_repo
from src.models import Item

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000"
    "00907753de0000000c4944415408d76360000000000004000146a13a"
    "020000000049454e44ae426082"
)

_FAKE_DESC = ItemDescription(
    object_class="umbrella", colors=["black"], confidence=0.85
)


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.save_image = MagicMock(return_value="safe_uuid.png")
    repo.insert_item = AsyncMock()
    repo.get_item = AsyncMock(return_value=None)
    repo.list_items = AsyncMock(return_value=[])
    repo.get_embeddings_by_status = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def client(mock_repo):
    app.dependency_overrides[get_repo] = lambda: mock_repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_wrong_mime_type_returns_400(client):
    """Uploading a non-image file must return 400, not a 500 crash."""
    response = client.post(
        "/items/lost",
        files={"file": ("resume.pdf", b"%PDF-1.4 content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_text_file_returns_400(client):
    response = client.post(
        "/items/lost",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_oversized_file_returns_413(client, monkeypatch):
    """A file larger than MAX_IMAGE_BYTES must return 413."""
    monkeypatch.setattr("src.api.settings.MAX_IMAGE_BYTES", 5)
    response = client.post(
        "/items/lost",
        files={"file": ("photo.png", b"\x89PNG" + b"\x00" * 100, "image/png")},
    )
    assert response.status_code == 413


def test_jpeg_is_accepted(client, mock_repo):
    """image/jpeg must be accepted (not rejected as unsupported)."""
    mock_repo.insert_item = AsyncMock(return_value=Item(
        id=1,
        status="lost",
        filename="x.jpg",
        created_at=datetime.now(UTC),
        vlm_description=_FAKE_DESC.model_dump(),
        embedding=[0.1],
    ))
    with patch("src.api.describe_item_async", AsyncMock(return_value=_FAKE_DESC)), \
         patch("src.api.embed_async", AsyncMock(return_value=np.array([0.1, 0.2], dtype=np.float32))):
        response = client.post(
            "/items/lost",
            files={"file": ("photo.jpg", _TINY_PNG, "image/jpeg")},
        )
    assert response.status_code != 400


def test_provider_error_returns_502(client):
    """A ProviderError after all retries must return HTTP 502."""
    with patch(
        "src.api.describe_item_async",
        AsyncMock(side_effect=ProviderError("provider down")),
    ):
        response = client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    assert response.status_code == 502
    assert "provider" in response.json()["detail"].lower()


def test_provider_error_response_contains_message(client):
    """The 502 response body must include a human-readable message."""
    with patch(
        "src.api.describe_item_async",
        AsyncMock(side_effect=ProviderError("upstream quota exceeded")),
    ):
        response = client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    body = response.json()
    assert "message" in body


def test_embed_failure_returns_502(client):
    """An embedding failure (after describe succeeds) must also return 502."""
    with patch("src.api.describe_item_async", AsyncMock(return_value=_FAKE_DESC)), \
         patch("src.api.embed_async", AsyncMock(side_effect=ProviderError("embed down"))):
        response = client.post(
            "/items/lost",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    assert response.status_code == 502


def test_get_matches_missing_item_returns_404(client, mock_repo):
    mock_repo.get_item = AsyncMock(return_value=None)
    response = client.get("/items/99999/matches")
    assert response.status_code == 404


def test_get_matches_item_without_embedding_returns_422(client, mock_repo):
    item = Item(
        id=1, status="lost", filename="x.png",
        original_name="x.png", embedding=None,
    )
    mock_repo.get_item = AsyncMock(return_value=item)
    response = client.get("/items/1/matches")
    assert response.status_code == 422


async def test_partial_batch_failure_does_not_crash():
    """In register_batch, if one item fails the others still succeed."""
    from src.concurrency.pipeline import run_with_semaphore

    call_count = 0

    async def _sometimes_fails():
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ProviderError("one bad item")
        return call_count

    factories = [lambda: _sometimes_fails() for _ in range(4)]
    results = await run_with_semaphore(factories)

    errors = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(errors) == 1
    assert len(successes) == 3
