"""SQLite-backed repository implementation using ``aiosqlite``.

This is the default storage backend for local development and the entire
automated test suite (via an in-memory database). It satisfies
:class:`AbstractRepository` so calling code can swap to PostgreSQL in
production without modification.

Typical usage::

    repo = SQLiteRepository("dev.db")
    repo = SQLiteRepository(":memory:")
    await repo.initialize()
    ...
    await repo.close()
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import aiosqlite
import numpy as np

from src.models import Item
from src.storage.base import AbstractRepository
from src.tracing import traced_db_op

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    status        TEXT    NOT NULL CHECK (status IN ('lost', 'found')),
    filename      TEXT    NOT NULL,
    original_name TEXT,
    user_text     TEXT,
    vlm_description TEXT,
    embedding     TEXT,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status ON items (status);
"""

_SELECT_COLS = (
    "id, status, filename, original_name, user_text, "
    "vlm_description, embedding, created_at"
)


class SQLiteRepository(AbstractRepository):
    """Async SQLite repository backed by ``aiosqlite``.

    Instantiate once per process (or once per test via a fixture). Always
    call :meth:`initialize` before use and :meth:`close` when finished.
    The constructor accepts either a raw filesystem path / ``":memory:"``
    or any SQLAlchemy-style SQLite URL — the prefix is normalised away.
    """

    _SQLITE_PREFIXES = ("sqlite+aiosqlite:///", "sqlite:///", "sqlite://")

    def __init__(self, db_url: str = "sqlite+aiosqlite:///./dev.db") -> None:
        """Store the resolved database path; defer the actual connection.

        Args:
            db_url: A raw path, ``":memory:"``, or any SQLAlchemy-style
                SQLite URL (``sqlite:///``, ``sqlite+aiosqlite:///``, …).
        """
        path = db_url
        for prefix in self._SQLITE_PREFIXES:
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        self._db_path = path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection, enable WAL mode, and create the schema."""
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        log.info("sqlite_repo.initialized", extra={"db": self._db_path})

    async def close(self) -> None:
        """Close the underlying connection if one was opened."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.info("sqlite_repo.closed")

    @traced_db_op("insert_item")
    async def insert_item(self, item: Item) -> Item:
        """Insert a new row and return the item with its assigned ID.

        Args:
            item: Domain model to persist. Its ``id`` and ``created_at``
                fields are ignored; the database supplies both.

        Returns:
            A copy of the input with ``id`` and ``created_at`` populated.
        """
        self._assert_open()
        now = datetime.now(UTC).isoformat()

        async with self._conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO items
                (status, filename, original_name, user_text,
                 vlm_description, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.status,
                item.filename,
                item.original_name,
                item.user_text,
                json.dumps(item.vlm_description) if item.vlm_description else None,
                json.dumps(item.embedding) if item.embedding else None,
                now,
            ),
        ) as cursor:
            new_id = cursor.lastrowid

        await self._conn.commit()  # type: ignore[union-attr]
        log.info("sqlite_repo.insert_item", extra={"id": new_id, "status": item.status})
        return item.model_copy(
            update={"id": new_id, "created_at": datetime.fromisoformat(now)}
        )

    @traced_db_op("get_item")
    async def get_item(self, item_id: int) -> Item | None:
        """Fetch a single item by primary key.

        Args:
            item_id: Database ID to retrieve.

        Returns:
            The matching :class:`Item`, or ``None`` if no row has that ID.
        """
        self._assert_open()
        async with self._conn.execute(  # type: ignore[union-attr]
            f"SELECT {_SELECT_COLS} FROM items WHERE id = ?", (item_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_item(tuple(row)) if row is not None else None

    @traced_db_op("list_items")
    async def list_items(self, status: str | None = None) -> list[Item]:
        """Return all items, optionally filtered to a specific status.

        Args:
            status: ``"lost"``, ``"found"``, ``"all"``, or ``None``.
                ``"all"`` and ``None`` both return every item.

        Returns:
            Items ordered by descending ID (newest first).
        """
        self._assert_open()
        if status and status != "all":
            query = (
                f"SELECT {_SELECT_COLS} FROM items WHERE status = ? ORDER BY id DESC"
            )
            params: tuple = (status,)
        else:
            query = f"SELECT {_SELECT_COLS} FROM items ORDER BY id DESC"
            params = ()

        async with self._conn.execute(query, params) as cursor:  # type: ignore[union-attr]
            rows = await cursor.fetchall()

        return [self._row_to_item(tuple(r)) for r in rows]

    @traced_db_op("get_embeddings_by_status")
    async def get_embeddings_by_status(
        self, status: str
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``(id, embedding)`` pairs for every item with the given status.

        Items lacking an embedding are silently skipped — they cannot
        contribute to similarity ranking.
        """
        self._assert_open()
        async with self._conn.execute(  # type: ignore[union-attr]
            "SELECT id, embedding FROM items WHERE status = ? AND embedding IS NOT NULL",
            (status,),
        ) as cursor:
            rows = await cursor.fetchall()

        result: list[tuple[int, np.ndarray]] = []
        for row_id, emb_json in rows:
            if emb_json:
                vec = np.array(json.loads(emb_json), dtype=np.float32)
                result.append((row_id, vec))
        return result

    def _assert_open(self) -> None:
        """Raise if :meth:`initialize` has not yet been called.

        Catches the common bug of forgetting to await ``initialize()`` and
        produces a friendlier message than the underlying ``AttributeError``.
        """
        if self._conn is None:
            raise RuntimeError(
                "Repository is not initialised. Call await repo.initialize() first."
            )

    @staticmethod
    def _row_to_item(row: tuple) -> Item:
        """Project a raw row tuple back into the :class:`Item` domain model."""
        (
            id_,
            status,
            filename,
            original_name,
            user_text,
            vlm_json,
            emb_json,
            created_at_str,
        ) = row
        return Item(
            id=id_,
            status=status,
            filename=filename,
            original_name=original_name,
            user_text=user_text,
            vlm_description=json.loads(vlm_json) if vlm_json else None,
            embedding=json.loads(emb_json) if emb_json else None,
            created_at=datetime.fromisoformat(created_at_str),
        )
