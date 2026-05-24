"""Tests for the multi-provider failover wrappers (src/core/failover.py).

Validates that when the primary provider raises ``ProviderError``, the call
is transparently retried on the secondary — and that if both fail, the
exception propagates correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.providers.base import ProviderError
from src.core.failover import FailoverEmbedder, FailoverVLM


def _vlm(response: str = '{"object_class": "bag", "colors": ["blue"], "confidence": 0.9}') -> MagicMock:
    v = MagicMock()
    v.describe = MagicMock(return_value=response)
    return v


def _embedder(vec: list[float] | None = None) -> MagicMock:
    e = MagicMock()
    e.dimension = 4
    e.embed = MagicMock(return_value=np.array(vec or [1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return e


def test_failover_vlm_uses_primary_when_healthy():
    primary = _vlm("primary_response")
    secondary = _vlm("secondary_response")
    failover = FailoverVLM([primary, secondary])
    result = failover.describe("img.png", "prompt")
    assert result == "primary_response"
    secondary.describe.assert_not_called()


def test_failover_vlm_switches_to_secondary_on_provider_error():
    primary = _vlm()
    primary.describe = MagicMock(side_effect=ProviderError("primary down"))
    secondary = _vlm("secondary_response")

    failover = FailoverVLM([primary, secondary])
    result = failover.describe("img.png", "prompt")

    assert result == "secondary_response"


def test_failover_vlm_raises_when_all_fail():
    primary = _vlm()
    primary.describe = MagicMock(side_effect=ProviderError("primary down"))
    secondary = _vlm()
    secondary.describe = MagicMock(side_effect=ProviderError("secondary down"))

    failover = FailoverVLM([primary, secondary])
    with pytest.raises(ProviderError, match="All 2"):
        failover.describe("img.png", "prompt")


def test_failover_vlm_rejects_empty_provider_list():
    with pytest.raises(ValueError, match="at least one"):
        FailoverVLM([])


def test_failover_vlm_tries_all_providers_in_order():
    providers = [_vlm() for _ in range(3)]
    for i, p in enumerate(providers[:-1]):
        p.describe = MagicMock(side_effect=ProviderError(f"provider {i} down"))
    providers[-1].describe = MagicMock(return_value="last_response")

    failover = FailoverVLM(providers)
    result = failover.describe("img.png", "prompt")
    assert result == "last_response"


def test_failover_vlm_passes_json_schema():
    primary = _vlm()
    schema = {"type": "object"}
    failover = FailoverVLM([primary])
    failover.describe("img.png", "prompt", json_schema=schema)
    primary.describe.assert_called_once_with("img.png", "prompt", json_schema=schema)


def test_failover_embedder_uses_primary_when_healthy():
    primary = _embedder([1.0, 0.0, 0.0, 0.0])
    secondary = _embedder([0.0, 1.0, 0.0, 0.0])
    failover = FailoverEmbedder([primary, secondary])

    vec = failover.embed("test text")
    assert np.allclose(vec, [1.0, 0.0, 0.0, 0.0])
    secondary.embed.assert_not_called()


def test_failover_embedder_switches_to_secondary_on_error():
    primary = _embedder()
    primary.embed = MagicMock(side_effect=ProviderError("primary down"))
    secondary = _embedder([0.5, 0.5, 0.5, 0.5])

    failover = FailoverEmbedder([primary, secondary])
    vec = failover.embed("text")
    assert vec is not None


def test_failover_embedder_raises_when_all_fail():
    primary = _embedder()
    primary.embed = MagicMock(side_effect=ProviderError("p1"))
    secondary = _embedder()
    secondary.embed = MagicMock(side_effect=ProviderError("p2"))

    failover = FailoverEmbedder([primary, secondary])
    with pytest.raises(ProviderError, match="All 2"):
        failover.embed("text")


def test_failover_embedder_rejects_empty_provider_list():
    with pytest.raises(ValueError, match="at least one"):
        FailoverEmbedder([])


def test_failover_embedder_dimension_from_primary():
    primary = _embedder()
    primary.dimension = 1024
    failover = FailoverEmbedder([primary])
    assert failover.dimension == 1024
