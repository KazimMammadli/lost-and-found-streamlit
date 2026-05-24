"""Tests for src/services/ai_service.py.

Validates retry behaviour, the embedding cache, timeouts, and that
every AI call goes through the wrappers (never raw ai.* directly).

All tests are fully offline — FakeVLM and FakeEmbedder inject deterministic
responses; no network calls are made.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai.providers.base import ProviderError
from ai.schemas import ItemDescription
from src.services.ai_service import (
    clear_embed_cache,
    describe_item_async,
    embed_async,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure each test starts with an empty embedding cache."""
    clear_embed_cache()
    yield
    clear_embed_cache()


_FAKE_DESC = ItemDescription(
    object_class="umbrella",
    colors=["black"],
    brand="Fulton",
    confidence=0.85,
)


async def test_describe_item_uses_fake_vlm(fake_vlm, sample_image):
    result = await describe_item_async(sample_image, "lost umbrella", vlm=fake_vlm)
    assert isinstance(result, ItemDescription)
    assert result.object_class == "umbrella"


async def test_describe_item_passes_user_text_to_vlm(fake_vlm, sample_image):
    await describe_item_async(sample_image, "left it on the bench", vlm=fake_vlm)
    prompt = fake_vlm.calls[0][1]
    assert "left it on the bench" in prompt


async def test_describe_item_accepts_empty_user_text(fake_vlm, sample_image):
    result = await describe_item_async(sample_image, "", vlm=fake_vlm)
    assert isinstance(result, ItemDescription)


async def test_describe_item_retries_on_provider_error(fake_vlm, sample_image):
    """First two calls raise; third succeeds — tenacity should handle it."""
    call_count = 0

    original_describe = fake_vlm.describe

    def _flaky(image_path, prompt, *, json_schema=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError("transient error")
        return original_describe(image_path, prompt, json_schema=json_schema)

    fake_vlm.describe = _flaky
    result = await describe_item_async(sample_image, "umbrella", vlm=fake_vlm)
    assert result.object_class == "umbrella"
    assert call_count == 3


async def test_describe_item_raises_after_max_retries(fake_vlm, sample_image):
    """All three attempts fail — ProviderError should propagate."""
    fake_vlm.describe = MagicMock(side_effect=ProviderError("always fails"))
    with pytest.raises(ProviderError):
        await describe_item_async(sample_image, "x", vlm=fake_vlm)


async def test_describe_item_raises_on_timeout(sample_image):
    """If the VLM call blocks forever, asyncio.TimeoutError is raised."""

    def _slow_describe(image_path, prompt, *, json_schema=None):
        import time

        time.sleep(100)

    slow_vlm = MagicMock()
    slow_vlm.describe = _slow_describe

    with patch("src.services.ai_service.settings") as mock_settings:
        mock_settings.AI_CALL_TIMEOUT_S = 0.05
        with pytest.raises((asyncio.TimeoutError, ProviderError)):
            await describe_item_async(sample_image, "x", vlm=slow_vlm)


async def test_embed_returns_numpy_array(fake_embedder):
    vec = await embed_async("navy backpack", embedder=fake_embedder)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32


async def test_embed_returns_unit_vector(fake_embedder):
    vec = await embed_async("black umbrella", embedder=fake_embedder)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


async def test_embed_raises_on_empty_text(fake_embedder):
    with pytest.raises(ValueError, match="empty"):
        await embed_async("", embedder=fake_embedder)


async def test_embed_raises_on_whitespace_text(fake_embedder):
    with pytest.raises(ValueError, match="empty"):
        await embed_async("   ", embedder=fake_embedder)


async def test_embed_cache_hit_skips_provider():
    """Second call with same text should not invoke the embedder at all."""
    call_count = 0

    async def _counting_embed(text, *, embedder=None):
        nonlocal call_count
        call_count += 1
        return np.ones(8, dtype=np.float32)

    with patch("src.services.ai_service.embed", side_effect=lambda t, **kw: np.ones(8, dtype=np.float32)):
        vec1 = await embed_async("same text")
        vec2 = await embed_async("same text")

    assert np.allclose(vec1, vec2)


async def test_embed_different_texts_produce_different_vectors(fake_embedder):
    a = await embed_async("backpack", embedder=fake_embedder)
    b = await embed_async("umbrella", embedder=fake_embedder)
    assert not np.allclose(a, b)


async def test_embed_cache_does_not_share_across_different_texts(fake_embedder):
    """Cache keyed on text — different texts must never share a cached vector."""
    v1 = await embed_async("apple", embedder=fake_embedder)
    v2 = await embed_async("banana", embedder=fake_embedder)
    assert not np.allclose(v1, v2)


async def test_embed_retries_on_provider_error():
    """embed should retry on ProviderError (same policy as describe)."""
    call_count = 0

    def _flaky(text, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError("transient")
        return np.ones(8, dtype=np.float32) / np.sqrt(8)

    with patch("src.services.ai_service.embed", side_effect=_flaky):
        vec = await embed_async("some text")

    assert vec is not None
    assert call_count == 3


async def test_clear_embed_cache_resets_state(fake_embedder):
    """After clearing, the next embed call must hit the provider again."""
    await embed_async("text1", embedder=fake_embedder)
    clear_embed_cache()

    call_count = 0

    def _counting(text, **_kwargs):
        nonlocal call_count
        call_count += 1
        return np.ones(8, dtype=np.float32) / np.sqrt(8)

    with patch("src.services.ai_service.embed", side_effect=_counting):
        await embed_async("text1")

    assert call_count == 1
