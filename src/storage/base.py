"""Abstract repository interface for the Lost & Found storage layer.

The SQLite implementation (development and tests) and the PostgreSQL
implementation (production) both satisfy :class:`AbstractRepository`.
FastAPI injects the chosen implementation via ``Depends(get_repo)`` so
tests can swap in an in-memory repository without monkey-patching.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config import settings

if TYPE_CHECKING:
    from src.models import Item


class AbstractRepository(ABC):
    """Contract for every storage backend.

    Concrete subclasses provide the async database methods. The synchronous
    :meth:`save_image` helper is implemented here once and shared across all
    subclasses because it only touches the filesystem.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create the schema and open database connections.

        Must be called once before any other repository method is used.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release database connections and any backing resources."""

    @abstractmethod
    async def insert_item(self, item: Item) -> Item:
        """Persist a new item and return it with its assigned primary key."""

    @abstractmethod
    async def get_item(self, item_id: int) -> Item | None:
        """Return the item with the given ID or ``None`` if it does not exist."""

    @abstractmethod
    async def list_items(self, status: str | None = None) -> list[Item]:
        """Return all items, optionally filtered by ``status``.

        Args:
            status: ``"lost"``, ``"found"``, or ``None`` / ``"all"``
                to return every item regardless of status.
        """

    @abstractmethod
    async def get_embeddings_by_status(
        self, status: str
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``(item_id, embedding_vector)`` pairs for items matching ``status``.

        Only items that have a stored embedding are returned. The matching
        pipeline uses this to assemble the candidate pool for similarity
        scoring without paying the cost of materialising full ``Item`` objects.
        """

    def save_image(self, data: bytes, original_name: str) -> str:
        """Write image bytes to disk under a UUID-derived filename.

        Using a UUID rather than the caller-supplied filename eliminates
        path-traversal risk; ``original_name`` is consulted only for its
        extension so the on-disk file keeps a sensible suffix.

        Args:
            data: Raw image bytes from an upload.
            original_name: Original filename supplied by the client.
                Only the suffix (``.png`` / ``.jpg`` / ``.jpeg``) is honoured;
                any other suffix is replaced with ``.png``.

        Returns:
            The UUID-based filename (``<hex>.<ext>``) relative to
            ``IMAGES_DIR``. Persist this value in the database and rebuild
            the full path as ``Path(settings.IMAGES_DIR) / filename`` when
            reading the file back.

        Raises:
            ValueError: If, after resolution, the destination path escapes
                the configured ``IMAGES_DIR``.
        """
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"

        safe_name = uuid.uuid4().hex + suffix

        storage_root = Path(settings.IMAGES_DIR).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)

        dest = (storage_root / safe_name).resolve()

        if not str(dest).startswith(str(storage_root)):
            raise ValueError(
                f"Generated path {dest} escapes storage directory {storage_root}"
            )

        dest.write_bytes(data)
        return safe_name
