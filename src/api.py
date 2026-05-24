"""FastAPI HTTP server for the Smart Lost & Found service.

Endpoints
---------
* ``POST /items/lost`` — register a lost item (image + optional text).
* ``POST /items/found`` — register a found item (image + optional text).
* ``GET  /items/{id}/matches`` — top-k matches from the opposite pool.
* ``GET  /items`` — list all items (filterable by status).
* ``POST /items/batch-lost`` — register multiple lost items concurrently.
* ``POST /items/batch-found`` — register multiple found items concurrently.
* ``GET  /health`` — liveness probe plus a database connectivity check.

Design notes
------------
* The repository is created once during the FastAPI lifespan and made
  available to every endpoint via the ``get_repo`` dependency. Tests
  override the dependency to inject an in-memory SQLite repository.
* Uploads are validated (MIME type plus byte size) *before* any AI call,
  so a bad upload never wastes an API token or a database round-trip.
* Stored filenames are never derived from user input; the repository
  generates a UUID-based name to eliminate path-traversal risk.
* Every AI interaction goes through ``src.services.ai_service``, so
  retries, timeouts, caching, rate limiting, and cost telemetry are
  applied uniformly to single-item and batch endpoints alike.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ai.providers.base import ProviderError
from src.config import settings
from src.core.matcher import find_matches
from src.models import (
    Item,
    ItemListResponse,
    ItemResponse,
    MatchListResponse,
    MatchResponse,
)
from src.services.ai_service import describe_item_async, embed_async
from src.storage import build_repo
from src.storage.base import AbstractRepository

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the repository on startup and close it on shutdown.

    The constructed repository is attached to ``app.state.repo`` so
    that :func:`get_repo` can return it from any request handler.
    """
    repo = build_repo()
    await repo.initialize()
    app.state.repo = repo
    log.info("api.startup", extra={"db": settings.DATABASE_URL})
    yield
    await repo.close()
    log.info("api.shutdown")


app = FastAPI(
    title="Smart Lost & Found",
    description=(
        "Register lost and found items via VLM descriptions and find "
        "matches by embedding similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_repo() -> AbstractRepository:
    """Dependency-injection accessor for the lifespan-initialised repository.

    Tests override this via ``app.dependency_overrides[get_repo]`` to
    inject a fake repository without touching the real database.
    """
    return app.state.repo


_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}


async def _read_and_validate_image(file: UploadFile) -> bytes:
    """Read the upload's bytes and validate its MIME type and size.

    Args:
        file: Incoming FastAPI ``UploadFile``.

    Returns:
        The fully-read image bytes.

    Raises:
        HTTPException: ``400`` for an unsupported content type;
            ``413`` for a file larger than ``MAX_IMAGE_BYTES``.
    """
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Only image/png and image/jpeg are accepted."
            ),
        )
    data = await file.read()
    if len(data) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(data)} bytes). "
                f"Maximum allowed size is {settings.MAX_IMAGE_BYTES} bytes."
            ),
        )
    return data


async def _register_item(
    file: UploadFile,
    user_text: str | None,
    status: Literal["lost", "found"],
    repo: AbstractRepository,
) -> ItemResponse:
    """Run the full single-item registration pipeline.

    The pipeline validates the upload, persists the image bytes,
    describes the image with the VLM, embeds the description text, and
    inserts the resulting :class:`Item` into the repository.

    Args:
        file: Uploaded image.
        user_text: Optional user description.
        status: ``"lost"`` or ``"found"``.
        repo: Repository for image storage and DB inserts.

    Returns:
        Public :class:`ItemResponse` for the newly registered item.
    """
    data = await _read_and_validate_image(file)

    filename = repo.save_image(data, file.filename or "upload.png")
    image_path = str(Path(settings.IMAGES_DIR) / filename)

    log.info(
        "api.register.start",
        extra={
            "status": status,
            "filename": filename,
            "user_text_len": len(user_text or ""),
        },
    )

    description = await describe_item_async(image_path, user_text or "")
    embedding_vec = await embed_async(description.to_search_text())

    item = Item(
        status=status,
        filename=filename,
        original_name=file.filename,
        user_text=user_text,
        vlm_description=description.to_dict(),
        embedding=embedding_vec.tolist(),
    )
    saved = await repo.insert_item(item)

    log.info(
        "api.register.ok",
        extra={"id": saved.id, "status": status, "class": description.object_class},
    )
    return ItemResponse.from_item(saved)


@app.exception_handler(ProviderError)
async def _provider_error_handler(request, exc: ProviderError) -> JSONResponse:
    """Translate an unrecoverable provider failure into HTTP 502."""
    log.error("api.provider_error", extra={"error": str(exc)})
    return JSONResponse(
        status_code=502,
        content={"detail": "AI provider error", "message": str(exc)},
    )


@app.exception_handler(asyncio.TimeoutError)
async def _timeout_handler(request, exc: asyncio.TimeoutError) -> JSONResponse:
    """Translate an AI call timeout into HTTP 504."""
    log.error("api.timeout", extra={"error": str(exc)})
    return JSONResponse(
        status_code=504,
        content={"detail": "AI provider timed out", "message": str(exc)},
    )


@app.post("/items/lost", response_model=ItemResponse, status_code=201)
async def post_lost(
    file: UploadFile = File(..., description="JPEG or PNG image of the lost item"),
    user_text: str | None = Form(None, description="Optional free-text description"),
    repo: AbstractRepository = Depends(get_repo),
) -> ItemResponse:
    """Register a lost item.

    Uploads the image, runs the VLM description pipeline, and stores
    the item with its embedding for future similarity matching.
    """
    return await _register_item(file, user_text, "lost", repo)


@app.post("/items/found", response_model=ItemResponse, status_code=201)
async def post_found(
    file: UploadFile = File(..., description="JPEG or PNG image of the found item"),
    user_text: str | None = Form(None, description="Optional free-text description"),
    repo: AbstractRepository = Depends(get_repo),
) -> ItemResponse:
    """Register a found item.

    Same pipeline as :func:`post_lost` but tags the item as ``found``.
    """
    return await _register_item(file, user_text, "found", repo)


@app.get("/items/{item_id}/matches", response_model=MatchListResponse)
async def get_matches(
    item_id: int,
    k: Annotated[int, Query(ge=1, le=20)] = 3,
    repo: AbstractRepository = Depends(get_repo),
) -> MatchListResponse:
    """Return the top-k matches for ``item_id`` from the opposite pool.

    A lost item is matched against found items, and vice versa. Each
    match record is hydrated with the candidate item's public details
    so a single API call returns everything the client needs to render.
    """
    item = await repo.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
    if item.embedding is None:
        raise HTTPException(
            status_code=422,
            detail=f"Item {item_id} has no embedding. Re-register the item.",
        )

    records = await find_matches(item, repo, k=k)

    match_responses: list[MatchResponse] = []
    for record in records:
        candidate = await repo.get_item(record.candidate_id)
        match_responses.append(
            MatchResponse(
                candidate_id=record.candidate_id,
                score=record.score,
                reason=record.reason,
                candidate=ItemResponse.from_item(candidate) if candidate else None,
            )
        )

    return MatchListResponse(
        query_id=item_id,
        matches=match_responses,
        count=len(match_responses),
    )


@app.get("/items", response_model=ItemListResponse)
async def list_items(
    status: Annotated[str | None, Query(pattern="^(lost|found|all)$")] = None,
    repo: AbstractRepository = Depends(get_repo),
) -> ItemListResponse:
    """List all registered items, optionally filtered by status.

    The ``status`` query parameter accepts ``lost``, ``found``, or
    ``all`` (the default when omitted).
    """
    items = await repo.list_items(status)
    return ItemListResponse(
        items=[ItemResponse.from_item(i) for i in items],
        count=len(items),
        status_filter=status,
    )


@app.get("/health")
async def health(repo: AbstractRepository = Depends(get_repo)) -> dict:
    """Liveness probe with a lightweight database connectivity check.

    Always returns HTTP 200 with a JSON body. The ``db`` field reports
    ``ok`` when a no-op repository query succeeds, ``error`` otherwise.
    """
    try:
        await repo.list_items("lost")
        db_status = "ok"
    except Exception as exc:
        log.error("api.health.db_error", extra={"error": str(exc)})
        db_status = "error"

    return {"status": "ok", "db": db_status}


@app.post("/items/batch-lost", response_model=list[ItemResponse], status_code=201)
async def post_batch_lost(
    files: list[UploadFile] = File(...),
    repo: AbstractRepository = Depends(get_repo),
) -> list[ItemResponse]:
    """Register multiple lost items concurrently.

    Each image is processed independently, bounded by
    ``SEMAPHORE_LIMIT``. An individual failure is logged and skipped;
    successful items are still returned.
    """
    return await _batch_register(files, "lost", repo)


@app.post("/items/batch-found", response_model=list[ItemResponse], status_code=201)
async def post_batch_found(
    files: list[UploadFile] = File(...),
    repo: AbstractRepository = Depends(get_repo),
) -> list[ItemResponse]:
    """Register multiple found items concurrently.

    See :func:`post_batch_lost` for behaviour details.
    """
    return await _batch_register(files, "found", repo)


async def _batch_register(
    files: list[UploadFile],
    status: Literal["lost", "found"],
    repo: AbstractRepository,
) -> list[ItemResponse]:
    """Drive :func:`_register_item` across many uploads concurrently.

    Args:
        files: Incoming upload list.
        status: ``"lost"`` or ``"found"``.
        repo: Repository used for image storage and DB inserts.

    Returns:
        Successfully registered items in input order. Failed items are
        omitted from the response and logged at ``WARNING``.
    """
    from src.concurrency.pipeline import run_with_semaphore

    async def _one(f: UploadFile) -> ItemResponse:
        """Register a single upload via :func:`_register_item`."""
        return await _register_item(f, None, status, repo)

    tasks: list[Callable[[], Coroutine[Any, Any, Any]]] = []
    for f in files:
        tasks.append(lambda f=f: _one(f))  # type: ignore[misc]
    results = await run_with_semaphore(tasks)

    responses: list[ItemResponse] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.warning(
                "api.batch_register.item_failed",
                extra={"index": i, "error": str(r), "type": type(r).__name__},
            )
        else:
            responses.append(r)
    return responses
