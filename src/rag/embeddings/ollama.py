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
from rag.modelprofile import resolve_profile


# Per-request input cap. Ollama embeds a list sequentially in one request, so a whole
# eval corpus (hundreds of ~2k-char docs) in a single call blows the client timeout —
# the timeout must bound one SLICE, not the whole workload.
_BATCH = 64


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._url = settings.ollama_url
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._instruction = settings.query_instruction
        self._profile = resolve_profile(settings.embed_model, settings.model_profile)

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for i in range(0, len(inputs), _BATCH):
            rows.extend(await self._embed_once(inputs[i : i + _BATCH]))
        return rows

    async def _embed_once(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                f"{self._url}/api/embed",
                json={"model": self._model, "input": inputs},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # str(ReadTimeout) is empty — the type name is the actual diagnosis.
            raise EmbeddingError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc

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
        """Doc side: formatted per the model's profile, identifiers excluded."""
        inputs = [format_document(doc.title, doc.content, self._profile) for doc in documents]
        return await self._embed(inputs)

    async def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Query side: prefixed per the model's profile — embedded in one request."""
        inputs = [format_query(q, self._instruction, self._profile) for q in queries]
        return await self._embed(inputs)
