"""Domain models and API schemas for Smart Lost & Found.

Two layers of types live here:

* **Domain models** (:class:`Item`, :class:`MatchRecord`) — internal
  representations used by the storage and service layers. They map 1:1
  to database rows.

* **API schemas** (:class:`ItemResponse`, :class:`MatchListResponse`,
  :class:`ItemListResponse`, :class:`MatchResponse`) — what the HTTP
  API and CLI expose to callers. They intentionally omit the raw
  embedding vector, which is large and not useful to API consumers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    """A registered lost or found item, as stored in the database.

    Attributes:
        id: Primary key; ``None`` for items not yet persisted.
        status: Either ``"lost"`` or ``"found"``.
        filename: UUID-based safe filename inside ``IMAGES_DIR``.
        original_name: The filename the user supplied at upload time.
        user_text: Optional free-text description from the user.
        vlm_description: Structured description produced by the VLM.
        embedding: Unit-normalised embedding vector used for similarity
            search; never exposed to API callers.
        created_at: UTC timestamp set by the database on insert.
    """

    id: int | None = None
    status: Literal["lost", "found"]
    filename: str = Field(description="UUID-based safe filename stored in IMAGES_DIR")
    original_name: str | None = Field(None, description="Original filename supplied by the user")
    user_text: str | None = Field(None, description="Free-text description from the user")
    vlm_description: dict | None = Field(
        None, description="Serialised ItemDescription from the VLM"
    )
    embedding: list[float] | None = Field(
        None, description="Embedding vector for similarity search (not exposed in API responses)"
    )
    created_at: datetime | None = None


class MatchRecord(BaseModel):
    """A single match result produced by the similarity layer.

    Attributes:
        candidate_id: Database ID of the matched item.
        score: Cosine similarity score in ``[0, 1]``.
        reason: Human-readable explanation derived from shared
            attributes (colours, brand, object class).
    """

    candidate_id: int = Field(description="Database ID of the matching item")
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score [0, 1]")
    reason: str = Field(default="", description="Human-readable explanation of the match")


class ItemResponse(BaseModel):
    """Public representation of a registered item.

    The embedding vector is deliberately excluded — it is large and not
    useful to API consumers.
    """

    id: int
    status: str
    original_name: str | None
    user_text: str | None
    vlm_description: dict | None
    created_at: datetime

    @classmethod
    def from_item(cls, item: Item) -> ItemResponse:
        """Project a domain :class:`Item` into the public response shape.

        Args:
            item: Persisted item retrieved from the repository.

        Returns:
            A serialisation-ready :class:`ItemResponse` with the embedding
            vector stripped out.
        """
        return cls(
            id=item.id,  # type: ignore[arg-type]
            status=item.status,
            original_name=item.original_name,
            user_text=item.user_text,
            vlm_description=item.vlm_description,
            created_at=item.created_at,  # type: ignore[arg-type]
        )


class MatchResponse(BaseModel):
    """Public representation of a single match result.

    The optional ``candidate`` field is populated when the API hydrates
    each match with the matched item's full public details.
    """

    candidate_id: int
    score: float
    reason: str
    candidate: ItemResponse | None = Field(
        None, description="Full details of the matching item (included when available)"
    )


class MatchListResponse(BaseModel):
    """Response body for ``GET /items/{id}/matches`` — a ranked match list."""

    query_id: int
    matches: list[MatchResponse]
    count: int


class ItemListResponse(BaseModel):
    """Response body for ``GET /items`` — a list of registered items."""

    items: list[ItemResponse]
    count: int
    status_filter: str | None = Field(None, description="The status filter applied, if any")
