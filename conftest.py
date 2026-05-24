"""Root pytest configuration.

Two responsibilities:

1. Adds the project root to :data:`sys.path` so ``from ai import ...``
   and ``from src import ...`` work in every test without requiring an
   editable install of the package.
2. Re-exports :class:`FakeVLM` and :class:`FakeEmbedder` from
   :mod:`tests.fakes` as project-wide pytest fixtures (``fake_vlm``,
   ``fake_embedder``) and supplies the shared ``sample_image`` fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.fakes import FakeEmbedder, FakeVLM  # noqa: E402  (after path setup)

__all__ = ["FakeVLM", "FakeEmbedder"]


@pytest.fixture
def fake_vlm() -> FakeVLM:
    """Return a :class:`FakeVLM` with the default canned payload."""
    return FakeVLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Return a :class:`FakeEmbedder` producing 8-D unit vectors."""
    return FakeEmbedder()


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    """Write a minimal valid 1×1 PNG to a temp path and return its location.

    The :class:`FakeVLM` ignores the contents, so a tiny synthetic file
    is enough to satisfy any test that needs a real path on disk.
    """
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(png_bytes)
    return str(img_path)
