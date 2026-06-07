"""Ollama-backed Embedder adapter (implements rag.core.ports.Embedder).

Holds its configuration and HTTP client (constructor-injected — no globals) and
applies the asymmetric formatting from rag.core. Transport/response problems are
translated into domain errors so the web layer can map them without knowing about
httpx.
"""
from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag.config import Settings
from rag.core.entities import Document
from rag.core.errors import EmbeddingError
from rag.core.formatting import format_document, format_query


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._url = settings.ollama_url
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._instruction = settings.query_instruction

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                f"{self._url}/api/embed",
                json={"model": self._model, "input": inputs},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama request failed: {exc}") from exc

        embeddings = resp.json().get("embeddings")
        if not embeddings:
            raise EmbeddingError("Ollama returned no embeddings")

        dim = len(embeddings[0])
        if dim != self._dim:
            raise EmbeddingError(
                f"Embedding dim {dim} != expected {self._dim}. "
                f"Set EMBED_DIM={dim} to match the model."
            )
        return embeddings

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Doc side: title prepended to body, identifiers excluded."""
        inputs = [format_document(doc.title, doc.content) for doc in documents]
        return await self._embed(inputs)

    async def embed_query(self, query: str) -> list[float]:
        """Query side: instruction-prefixed."""
        embeddings = await self._embed([format_query(query, self._instruction)])
        return embeddings[0]
