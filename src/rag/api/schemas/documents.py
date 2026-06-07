"""Request/response models for the /documents endpoint."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    content: str
    # The page URL is the base unit's identifier. domain/path are derived from it
    # server-side (rag.core.urls.build_document_metadata) and need not be sent.
    url: str | None = None
    title: str | None = None
    # Optional caller-supplied extras, merged with the derived url/domain/path/title.
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentsRequest(BaseModel):
    documents: list[DocumentIn]


class DocumentsResponse(BaseModel):
    ids: list[int]
    count: int
