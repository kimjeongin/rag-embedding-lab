"""SentenceTransformer-backed Embedder adapter (implements rag.core.ports.Embedder).

Embeds with a LOCAL sentence-transformers model — e.g. one fine-tuned by
`rag.training` (saved under outputs/...). It's the alternative to OllamaEmbedder:
selecting it (EMBEDDER=sentence-transformers) evaluates the fine-tuned model instead of
the Ollama one — only the place that builds the embedder picks which one.

Applies the same rag.core.formatting as serving/training (parity). Needs the
training stack (`uv sync --group training`); the heavy imports are deferred to
__init__ so importing this module stays light.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rag.config import Settings
from rag.core.entities import Document
from rag.core.errors import EmbeddingError
from rag.core.formatting import format_document, format_query


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise EmbeddingError(
                "sentence-transformers is not installed — run `uv sync --group training`"
            ) from exc

        device = settings.st_device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model = SentenceTransformer(settings.st_model, device=device)
        self._instruction = settings.query_instruction

        dim = self._model.get_sentence_embedding_dimension()
        if dim != settings.embed_dim:
            raise EmbeddingError(
                f"Model '{settings.st_model}' outputs dim {dim} != EMBED_DIM "
                f"{settings.embed_dim}. Set EMBED_DIM={dim} to match the model."
            )

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Doc side: title prepended to body, identifiers excluded."""
        inputs = [format_document(doc.title, doc.content) for doc in documents]
        return await self._encode(inputs)

    async def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Query side: instruction-prefixed — encoded in one batch."""
        inputs = [format_query(q, self._instruction) for q in queries]
        return await self._encode(inputs)

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        # encode() is blocking (torch); run it off the event loop.
        try:
            vectors = await asyncio.to_thread(
                self._model.encode, texts, normalize_embeddings=True
            )
        except Exception as exc:  # noqa: BLE001 - surface any encode failure uniformly
            raise EmbeddingError(f"sentence-transformers encode failed: {exc}") from exc
        return [vector.tolist() for vector in vectors]
