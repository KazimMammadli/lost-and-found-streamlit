"""Smoke tests for the SE layer scaffolding.

These tests verify that all major modules can be imported without errors and
that the public API surface matches expectations.  They complement the
deeper unit tests in the other test_*.py files.
"""

from __future__ import annotations

from datetime import UTC

import pytest


def test_src_imports():
    import src

    assert src is not None


def test_config_loads():
    from src.config import settings

    assert settings.IMAGES_DIR is not None
    assert settings.MAX_IMAGE_BYTES > 0
    assert settings.SEMAPHORE_LIMIT > 0


def test_models_importable():
    from src.models import Item, ItemResponse, MatchRecord

    assert Item is not None
    assert MatchRecord is not None
    assert ItemResponse is not None


def test_storage_factory_importable():
    from src.storage import build_repo

    assert callable(build_repo)


def test_api_importable():
    from src.api import app

    assert app is not None


def test_cli_importable():
    from src.cli import main

    assert callable(main)


def test_ai_module_importable():
    from ai import cosine, describe_item, embed, top_k

    assert callable(describe_item)
    assert callable(embed)
    assert callable(cosine)
    assert callable(top_k)


def test_item_model_validates_status():
    from pydantic import ValidationError

    from src.models import Item

    with pytest.raises(ValidationError):
        Item(status="invalid_status", filename="x.png")


def test_item_response_from_item():
    from datetime import datetime

    from src.models import Item, ItemResponse

    item = Item(
        id=5,
        status="lost",
        filename="abc.png",
        original_name="photo.png",
        vlm_description={"object_class": "bag", "colors": ["red"], "confidence": 0.9},
        embedding=[0.1, 0.2],
        created_at=datetime.now(UTC),
    )
    resp = ItemResponse.from_item(item)
    assert resp.id == 5
    assert resp.status == "lost"
    assert not hasattr(resp, "embedding") or "embedding" not in resp.model_fields
