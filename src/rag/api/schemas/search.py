"""Request/response DTOs for /search, with mappers from domain entities.

The `from_*` classmethods are the single place the web layer translates domain
entities (Hit/Site) into wire DTOs, keeping that mapping out of the route bodies.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.core.entities import Hit, Site


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    mode: Literal["page", "site"] = "page"
    # page mode: cap results per domain for diversity (None = pure similarity order).
    max_per_domain: int | None = Field(default=None, ge=1)
    # site mode (and page mode when max_per_domain is set): candidate pool size.
    fetch_k: int = Field(default=50, ge=1, le=500)


# --- page mode --------------------------------------------------------------
class PageHit(BaseModel):
    id: int
    url: str | None = None
    title: str | None = None
    domain: str | None = None
    content: str
    similarity: float

    @classmethod
    def from_hit(cls, hit: Hit) -> "PageHit":
        return cls(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            domain=hit.domain,
            content=hit.content,
            similarity=hit.similarity,
        )


class PageSearchResponse(BaseModel):
    mode: Literal["page"] = "page"
    query: str
    top_k: int
    results: list[PageHit]


# --- site mode --------------------------------------------------------------
class SitePage(BaseModel):
    url: str | None = None
    title: str | None = None
    similarity: float

    @classmethod
    def from_hit(cls, hit: Hit) -> "SitePage":
        return cls(url=hit.url, title=hit.title, similarity=hit.similarity)


class SiteHit(BaseModel):
    domain: str | None = None
    site_score: float
    pages: list[SitePage]

    @classmethod
    def from_site(cls, site: Site) -> "SiteHit":
        return cls(
            domain=site.domain,
            site_score=site.score,
            pages=[SitePage.from_hit(page) for page in site.pages],
        )


class SiteSearchResponse(BaseModel):
    mode: Literal["site"] = "site"
    query: str
    top_k: int
    fetch_k: int
    results: list[SiteHit]
