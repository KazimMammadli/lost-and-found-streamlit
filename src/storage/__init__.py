"""Storage layer package.

Exports the abstract repository interface and the :func:`build_repo`
factory, which selects the appropriate concrete implementation by
inspecting ``settings.DATABASE_URL``.

The HTTP API resolves the repository via the ``get_repo`` dependency in
``src/api.py``; the CLI calls :func:`build_repo` directly.
"""

from src.storage.base import AbstractRepository
from src.storage.sqlite_repo import SQLiteRepository


def build_repo() -> AbstractRepository:
    """Instantiate the repository implementation matching ``DATABASE_URL``.

    Returns:
        :class:`PostgreSQLRepository` when ``DATABASE_URL`` starts with
        ``postgresql``; :class:`SQLiteRepository` otherwise.
    """
    from src.config import settings

    url = settings.DATABASE_URL
    if url.startswith("postgresql"):
        from src.storage.repository import PostgreSQLRepository

        return PostgreSQLRepository(url)

    return SQLiteRepository(url)


__all__ = ["AbstractRepository", "SQLiteRepository", "build_repo"]
