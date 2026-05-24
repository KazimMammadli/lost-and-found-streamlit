"""End-to-end smoke test mirroring ``scripts/demo.py``.

Exercises the full offline pipeline — image bytes -> VLM description ->
embedding -> repository -> match retrieval — using the in-memory SQLite
backend and the canonical :class:`FakeVLM` / :class:`FakeEmbedder`. This
acts as a regression guard against breakage of the integration surface
that the demo and the API rely on.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from src.core.matcher import find_matches
from src.models import Item
from src.services.ai_service import (
    clear_embed_cache,
    describe_item_async,
    embed_async,
)
from src.storage.sqlite_repo import SQLiteRepository
from tests.fakes import FakeEmbedder, FakeVLM


def _make_png(color: tuple[int, int, int]) -> bytes:
    """Return raw bytes of a 2x2 solid-colour PNG (no Pillow dependency)."""
    width = height = 2
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    """Register a lost and a found item, then verify a match is returned."""
    clear_embed_cache()

    repo = SQLiteRepository(":memory:")
    await repo.initialize()
    try:
        vlm = FakeVLM()
        embedder = FakeEmbedder()

        lost_bytes = _make_png((200, 50, 50))
        lost_name = repo.save_image(lost_bytes, "lost.png")
        lost_desc = await describe_item_async(
            str(Path("data/images") / lost_name), "red backpack", vlm=vlm
        )
        lost_vec = await embed_async(lost_desc.to_search_text(), embedder=embedder)
        lost = await repo.insert_item(
            Item(
                status="lost",
                filename=lost_name,
                original_name="lost.png",
                user_text="red backpack",
                vlm_description=lost_desc.to_dict(),
                embedding=lost_vec.tolist(),
            )
        )
        assert lost.id is not None

        found_bytes = _make_png((200, 50, 50))
        found_name = repo.save_image(found_bytes, "found.png")
        found_desc = await describe_item_async(
            str(Path("data/images") / found_name), "red backpack", vlm=vlm
        )
        found_vec = await embed_async(found_desc.to_search_text(), embedder=embedder)
        found = await repo.insert_item(
            Item(
                status="found",
                filename=found_name,
                original_name="found.png",
                user_text="red backpack",
                vlm_description=found_desc.to_dict(),
                embedding=found_vec.tolist(),
            )
        )
        assert found.id is not None

        matches = await find_matches(lost, repo, k=3)
        assert len(matches) >= 1
        assert matches[0].candidate_id == found.id
        assert 0.0 <= matches[0].score <= 1.0

        items = await repo.list_items("all")
        assert len(items) == 2
    finally:
        await repo.close()
