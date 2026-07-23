"""SentenceTransformer-backed Embedder adapter (implements rag.core.ports.Embedder).

Embeds with a LOCAL sentence-transformers model — e.g. one fine-tuned by
`rag.training` (saved under outputs/...). This is the DEFAULT backend (the lab's own
path: train/eval/serve all load the model in-process); OllamaEmbedder is the
EMBEDDER=ollama alternative for parity checks — only the place that builds the
embedder picks which one.

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
from rag.modelprofile import resolve_profile


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
        # truncate_dim (Matryoshka inference): ST truncates each embedding to the first
        # N dims; normalize_embeddings then re-normalizes the prefix. get_sentence_
        # embedding_dimension() reports N, so the dim guard below still holds.
        self._model = SentenceTransformer(
            settings.st_model, device=device, truncate_dim=settings.truncate_dim
        )
        self._instruction = settings.query_instruction
        self._profile = resolve_profile(settings.st_model, settings.model_profile)

        # We prepend the profile's prefixes ourselves and call encode() (not
        # encode_query/encode_document), so a model that also applies a prompt BY
        # DEFAULT would double-prefix — "query: query: …" — and only look slightly
        # worse. Models ship default_prompt_name=null today; refuse if one doesn't.
        if getattr(self._model, "default_prompt_name", None):
            raise EmbeddingError(
                f"'{settings.st_model}'은 default_prompt_name="
                f"{self._model.default_prompt_name!r}을 갖고 있어 랩이 붙이는 "
                f"'{self._profile.name}' 접두사와 이중 적용됩니다 — 모델의 "
                f"config_sentence_transformers.json에서 default_prompt_name을 null로 "
                f"두거나 MODEL_PROFILE=plain으로 랩 쪽 접두사를 끄세요"
            )

        dim = self._model.get_embedding_dimension()
        if dim != settings.embed_dim:
            raise EmbeddingError(
                f"Model '{settings.st_model}' outputs dim {dim} != EMBED_DIM "
                f"{settings.embed_dim}. Set EMBED_DIM={dim} to match the model."
            )

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Doc side: formatted per the model's profile, identifiers excluded."""
        inputs = [format_document(doc.title, doc.content, self._profile) for doc in documents]
        return await self._encode(inputs)

    async def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Query side: prefixed per the model's profile — encoded in one batch."""
        inputs = [format_query(q, self._instruction, self._profile) for q in queries]
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
