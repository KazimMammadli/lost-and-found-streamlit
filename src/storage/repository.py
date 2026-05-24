"""PostgreSQL-backed repository implementation using ``asyncpg``.

This backend is intended for production deployments via docker-compose
where a real PostgreSQL server is available. Connection strings must use
asyncpg's native ``postgresql://`` scheme; the SQLAlchemy-style
``postgresql+asyncpg://`` form is also accepted and silently normalised.

Typical usage::

    repo = PostgreSQLRepository("postgresql://user:pass@host:5432/db")
    await repo.initialize()
    ...
    await repo.close()
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import asyncpg
import numpy as np

from src.models import Item
from src.storage.base import AbstractRepository
from src.tracing import traced_db_op

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS items (
    id            SERIAL PRIMARY KEY,
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


class PostgreSQLRepository(AbstractRepository):
    """Async PostgreSQL repository backed by an ``asyncpg`` connection pool.

    asyncpg is a high-performance native PostgreSQL client. Its parameter
    placeholders are ``$1, $2, …`` rather than the ``?`` used by SQLite —
    the only meaningful difference between this implementation and
    :class:`SQLiteRepository`.
    """

    def __init__(self, db_url: str) -> None:
        """Normalise the DSN and defer pool creation until :meth:`initialize`.

        Args:
            db_url: Connection string. SQLAlchemy-style
                ``postgresql+asyncpg://`` URLs are rewritten to the plain
                ``postgresql://`` form asyncpg expects.
        """
        self._dsn = db_url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create the connection pool and apply the schema DDL.

        The pool keeps a minimum of two warm connections and grows up to
        ten under load.
        """
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(_DDL)
        log.info("postgres_repo.initialized", extra={"dsn": self._dsn.split("@")[-1]})

    async def close(self) -> None:
        """Gracefully drain and close every pooled connection."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("postgres_repo.closed")

    @traced_db_op("insert_item")
    async def insert_item(self, item: Item) -> Item:
        """Insert a new row and return the item with its assigned ID.

        Uses ``RETURNING`` so the new row's ID and ``created_at`` come
        back in the same round-trip as the insert.
        """
        self._assert_open()
        now = datetime.now(UTC).isoformat()

        row = await self._pool.fetchrow(  # type: ignore[union-attr]
            f"""
            INSERT INTO items
                (status, filename, original_name, user_text,
                 vlm_description, embedding, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING {_SELECT_COLS}
            """,
            item.status,
            item.filename,
            item.original_name,
            item.user_text,
            json.dumps(item.vlm_description) if item.vlm_description else None,
            json.dumps(item.embedding) if item.embedding else None,
            now,
        )
        result = self._row_to_item(row)
        log.info("postgres_repo.insert_item", extra={"id": result.id, "status": item.status})
        return result

    @traced_db_op("get_item")
    async def get_item(self, item_id: int) -> Item | None:
        """Fetch a single item by primary key or return ``None``."""
        self._assert_open()
        row = await self._pool.fetchrow(  # type: ignore[union-attr]
            f"SELECT {_SELECT_COLS} FROM items WHERE id = $1", item_id
        )
        return self._row_to_item(row) if row is not None else None

    @traced_db_op("list_items")
    async def list_items(self, status: str | None = None) -> list[Item]:
        """Return all items, optionally filtered to a specific status."""
        self._assert_open()
        if status and status != "all":
            rows = await self._pool.fetch(  # type: ignore[union-attr]
                f"SELECT {_SELECT_COLS} FROM items WHERE status = $1 ORDER BY id DESC",
                status,
            )
        else:
            rows = await self._pool.fetch(  # type: ignore[union-attr]
                f"SELECT {_SELECT_COLS} FROM items ORDER BY id DESC"
            )
        return [self._row_to_item(r) for r in rows]

    @traced_db_op("get_embeddings_by_status")
    async def get_embeddings_by_status(
        self, status: str
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``(id, embedding)`` pairs for items matching ``status``.

        Items without a stored embedding are silently skipped.
        """
        self._assert_open()
        rows = await self._pool.fetch(  # type: ignore[union-attr]
            "SELECT id, embedding FROM items WHERE status = $1 AND embedding IS NOT NULL",
            status,
        )
        result: list[tuple[int, np.ndarray]] = []
        for row in rows:
            if row["embedding"]:
                vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
                result.append((row["id"], vec))
        return result

    def _assert_open(self) -> None:
        """Raise if :meth:`initialize` has not yet been called."""
        if self._pool is None:
            raise RuntimeError(
                "Repository is not initialised. Call await repo.initialize() first."
            )

    @staticmethod
    def _row_to_item(row: asyncpg.Record) -> Item:
        """Project an :class:`asyncpg.Record` back into an :class:`Item`."""
        return Item(
            id=row["id"],
            status=row["status"],
            filename=row["filename"],
            original_name=row["original_name"],
            user_text=row["user_text"],
            vlm_description=json.loads(row["vlm_description"])
            if row["vlm_description"]
            else None,
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
