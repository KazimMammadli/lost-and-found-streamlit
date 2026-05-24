"""Fake AI providers for offline testing and benchmarking.

Both classes satisfy the same abstract interfaces as the real providers
(:class:`VLMProvider`, :class:`EmbeddingProvider`) so they can be
injected anywhere a real provider is expected without monkey-patching.

Importing from here is the canonical way to obtain fake providers in:

* pytest fixtures (via ``conftest.py``)
* offline scripts (``scripts/bench.py``, ``scripts/demo.py``)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ai.providers.base import EmbeddingProvider, VLMProvider


class FakeVLM(VLMProvider):
    """VLM provider that returns a fixed JSON response without network I/O.

    Each call is recorded on :attr:`calls` so tests can assert on the
    inputs the provider received.
    """

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        """Bind the canned payload returned by every :meth:`describe` call.

        Args:
            payload: Optional override of the default structured
                description. Must conform to ``ITEM_DESCRIPTION_SCHEMA``
                (``object_class``, ``colors``, ``confidence`` are required).
        """
        self.payload: dict[str, Any] = payload or {
            "object_class": "umbrella",
            "colors": ["black"],
            "brand": "Fulton",
            "distinguishing_marks": ["bent rib"],
            "location_hints": ["library entrance"],
            "confidence": 0.85,
        }
        self.calls: list[tuple[str, str]] = []

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Record the call and return :attr:`payload` as a JSON string."""
        self.calls.append((image_path, prompt))
        return json.dumps(self.payload)


class FakeEmbedder(EmbeddingProvider):
    """Deterministic toy embedder producing unit vectors from a hash seed.

    The same input always produces the same vector and different inputs
    produce different (but stable) vectors. No network access required,
    so the embedder is safe for unit tests and offline benchmarks.
    """

    def __init__(self, dim: int = 8) -> None:
        """Bind the embedding dimensionality.

        Args:
            dim: Number of dimensions in the produced vectors. Defaults
                to 8 — large enough to keep similarity comparisons useful
                while small enough to remain cheap.
        """
        self._dim = dim

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised vector deterministically derived from ``text``.

        Args:
            text: Non-empty input string.

        Returns:
            ``float32`` unit vector of length :attr:`dimension`.

        Raises:
            ValueError: ``text`` is empty or whitespace-only.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v
