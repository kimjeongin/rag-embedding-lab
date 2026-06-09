"""Domain entities — the typed values that cross layer boundaries.

Plain frozen dataclasses with no dependency on FastAPI, httpx, or config. ``Document``
is the unit the embedder consumes: its title + content drive the asymmetric document
formatting in ``rag.core.formatting`` (the same formatting training uses, for parity).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """A page/passage to embed — only the fields the embedder actually reads."""

    content: str
    title: str | None = None
