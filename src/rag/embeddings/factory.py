"""Embedder factory — builds the embedder selected by ``Settings.embedder``.

The "which backend" decision lives in exactly ONE place, shared by the API
composition root and the CLI tools (rag-eval, rag-gen-synthetic). It's an async
context manager so the Ollama HTTP client's lifecycle is handled for the caller.
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import httpx

from rag.config import Settings
from rag.core.ports import Embedder
from rag.embeddings.ollama import OllamaEmbedder
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder


@contextlib.asynccontextmanager
async def build_embedder(settings: Settings) -> AsyncIterator[Embedder]:
    """Yield the configured Embedder, managing any resources it needs."""
    if settings.embedder == "ollama":
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as http:
            yield OllamaEmbedder(http, settings)
    elif settings.embedder == "sentence-transformers":
        yield SentenceTransformerEmbedder(settings)
    else:
        raise ValueError(
            f"unknown EMBEDDER {settings.embedder!r} (use 'ollama' or 'sentence-transformers')"
        )
