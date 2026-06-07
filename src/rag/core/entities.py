"""Domain entities — the typed values that cross layer boundaries.

Plain frozen dataclasses with no dependency on FastAPI, qdrant_client, httpx, or config.
These replace the stringly-typed dicts that used to flow between retrieval and the
API, so every layer speaks the same explicit vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A page to index (the input to the indexing use case)."""

    content: str
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    """A document ready to persist: raw body + derived metadata + its vector."""

    content: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class Hit:
    """A page retrieved from the store, with its similarity score.

    url/title/domain are surfaced from the stored metadata for grouping and for
    the response DTOs. (The full metadata stays in the store; the search path only
    needs these fields.)
    """

    id: int
    content: str
    similarity: float
    url: str | None = None
    title: str | None = None
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class Site:
    """A domain-grouped set of hits with an aggregate score (site mode)."""

    domain: str | None
    score: float
    pages: list[Hit]
