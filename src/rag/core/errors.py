"""Domain exceptions.

Raised by adapters in domain terms so the web layer can map them to an HTTP status
in one place (rag.api.errors), instead of routes catching transport-specific
exceptions like httpx.HTTPError.
"""
from __future__ import annotations


class EmbeddingError(Exception):
    """The embedding service failed or returned an unusable response."""
