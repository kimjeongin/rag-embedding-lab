"""Judging-loop ranking with a warm corpus cache — embed the corpus once, judge many.

``/data/label/search`` ranks the WHOLE eval corpus for one human query. Doing that
from scratch per judgment — a ~GB model load plus an embedding pass over every
document — costs minutes on a laptop device, and the judging loop fires one query
at a time, many times ("하루 10개씩"). So one entry is cached per (backend, model,
dim, corpus content): the normalized doc matrix and the embedder that produced it.
A repeat judgment embeds ONE query and does one matmul (~ms).

Only one entry is kept: model weights are ~GB, so switching models replaces the
entry (closing its resources) instead of accumulating. When the requested stack is
the process's own serving embedder, that instance is borrowed instead of loading a
second copy of the same weights.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from rag.config import Settings
from rag.core.entities import Document
from rag.core.ports import Embedder

CacheKey = tuple[str, str, int, str]  # (backend, model, dim, corpus fingerprint)


@dataclass
class _Entry:
    key: CacheKey
    embedder: Embedder
    aclose: Callable[[], Awaitable[None]] | None  # None = borrowed, don't close
    doc_ids: list[str]
    matrix: object  # np.ndarray — typed loosely so importing this module stays light


_entry: _Entry | None = None
_lock = asyncio.Lock()  # one build at a time; concurrent judgments serialize


def map_hits_to_corpus_ids(
    hits: list[dict], corpus: dict[str, dict[str, str | None]], top_n: int = 10
) -> list[str]:
    """Map serving-index hits back to eval-corpus doc ids by exact content.

    The Qdrant payload carries (title, content) but not the eval corpus's doc ids —
    when the eval corpus was cut from the same crawl, the texts are identical, so a
    content match recovers the id qrels needs. Hits whose content isn't in the eval
    corpus (serving corpus drifted / superset) are skipped; an empty result tells
    the caller the index can't answer for this corpus."""
    by_content = {
        ((doc["title"] or ""), (doc["text"] or "")): doc_id for doc_id, doc in corpus.items()
    }
    ranked: list[str] = []
    for hit in hits:
        doc_id = by_content.get(((hit.get("title") or ""), (hit.get("content") or "")))
        if doc_id is not None and doc_id not in ranked:
            ranked.append(doc_id)
        if len(ranked) >= top_n:
            break
    return ranked


def corpus_fingerprint(eval_dir: str) -> str:
    """Content hash of corpus.jsonl — regenerating the corpus must invalidate the
    cached matrix (same reason eval runs carry an eval-set fingerprint)."""
    return hashlib.sha256((Path(eval_dir) / "corpus.jsonl").read_bytes()).hexdigest()[:12]


def _construct(settings: Settings) -> tuple[Embedder, Callable[[], Awaitable[None]]]:
    """Build an embedder OUTSIDE the usual context manager — the cache owns its
    lifetime across requests, so cleanup happens on eviction, not scope exit."""
    if settings.embedder == "ollama":
        import httpx

        from rag.embeddings.ollama import OllamaEmbedder

        http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return OllamaEmbedder(http, settings), http.aclose
    from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

    async def _noop() -> None:
        return None

    return SentenceTransformerEmbedder(settings), _noop


async def _evict() -> None:
    global _entry
    if _entry is not None and _entry.aclose is not None:
        await _entry.aclose()
    _entry = None


async def rank(
    settings: Settings,
    corpus: dict[str, dict[str, str | None]],
    corpus_fp: str,
    query: str,
    top_n: int = 10,
    shared: Embedder | None = None,
) -> list[str]:
    """Top-N doc ids for one judging query, reusing the cached corpus matrix.

    ``settings`` selects the stack (lab.build_eval_settings output); ``shared`` is
    the process serving embedder when it IS that stack — borrowed, never closed.
    """
    global _entry
    import numpy as np

    from rag.evaluation.retrieval import l2_normalize

    key: CacheKey = (settings.embedder, settings.active_model, settings.embed_dim, corpus_fp)
    async with _lock:
        if _entry is None or _entry.key != key:
            await _evict()
            embedder, aclose = (shared, None) if shared is not None else _construct(settings)
            doc_ids = list(corpus)
            docs = [
                Document(content=corpus[d]["text"] or "", title=corpus[d]["title"])
                for d in doc_ids
            ]
            matrix = l2_normalize(
                np.asarray(await embedder.embed_documents(docs), dtype="float32")
            )
            _entry = _Entry(key, embedder, aclose, doc_ids, matrix)

        q = l2_normalize(np.asarray(await _entry.embedder.embed_queries([query]), dtype="float32"))
        sims = (q @ _entry.matrix.T)[0]
        order = np.argsort(-sims)[:top_n]
        return [_entry.doc_ids[i] for i in order]
