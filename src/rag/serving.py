"""Serving flow — build the Qdrant index from a corpus, and search it.

This is the lab's "actually serve it" side: the crawled corpus (page-level records,
the production index unit) is embedded with the SAME formatting as training/eval
(rag.core.formatting via the Embedder port) and upserted into Qdrant.

**Model change = full reindex, automated.** Vectors from different models don't share
a space, so every (model, dim, corpus) combination gets its OWN versioned collection —
the name encodes all three:

    {prefix}__{model-slug}__{dim}d__{corpus-fingerprint}

Search never uses those names; it queries the stable alias ``{prefix}-live``.
``index_corpus`` builds the versioned collection (skipping the embed work when it
already exists and is fully populated — rerunning is a cheap no-op) and atomically
repoints the alias. Old collections stay for instant rollback until ``prune`` drops
them. Framework-free (no fastapi): the CLI and the API routes both drive this.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Callable

from rag.config import Settings
from rag.core.entities import Document
from rag.core.errors import VectorStoreError
from rag.core.ports import Embedder
from rag.dataset import file_fingerprint, load_jsonl
from rag.vectorstore.qdrant import QdrantStore

_EMBED_BATCH = 32


def live_alias(prefix: str) -> str:
    """The stable serving pointer for a collection family."""
    return f"{prefix}-live"


def _slug(text: str) -> str:
    """Filesystem-path/model-name → collection-name-safe token."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def collection_name(prefix: str, model: str, dim: int, fingerprint: str | None) -> str:
    """The versioned collection for one (model, dim, corpus) — same inputs, same name,
    which is what makes reindexing idempotent."""
    return f"{prefix}__{_slug(model)}__{dim}d__{fingerprint or 'nofp'}"


def point_id(key: str) -> str:
    """Deterministic UUID for a document key (url — the crawl's dedupe identity), so a
    re-run upserts over the same points instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _doc_key(record: dict) -> str:
    return record.get("url") or record.get("_id") or f"{record.get('title')}\x00{record.get('content')}"


async def index_corpus(
    settings: Settings,
    embedder: Embedder,
    store: QdrantStore,
    corpus_file: str,
    *,
    recreate: bool = False,
    batch_size: int = _EMBED_BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Embed ``corpus_file`` into the versioned collection and repoint the live alias.

    Idempotent: if the target collection already holds every record (same model, dim
    and corpus content — all encoded in its name), the embed work is skipped and only
    the alias is ensured. ``recreate`` forces a rebuild of the same collection.
    """
    records = list(load_jsonl(corpus_file))
    if not records:
        raise VectorStoreError(f"코퍼스가 비어 있습니다: {corpus_file} — 먼저 `make crawl`로 수집하세요")

    prefix = settings.qdrant_collection
    alias = live_alias(prefix)
    name = collection_name(
        prefix, settings.active_model, settings.embed_dim, file_fingerprint(corpus_file)
    )

    existing = store.collection_info(name)
    if existing and recreate:
        store.delete_collection(name)
        existing = None
    if existing and existing["points"] >= len(records):
        store.swap_alias(alias, name)  # ensure the pointer, even if a prior run died before the swap
        return {"collection": name, "alias": alias, "docs": len(records),
                "dim": settings.embed_dim, "model": settings.active_model, "skipped": True}

    if not existing:
        store.create_collection(name, settings.embed_dim)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        docs = [Document(content=r.get("content") or r.get("text") or "", title=r.get("title"))
                for r in batch]
        vectors = await embedder.embed_documents(docs)
        store.upsert(name, [
            {
                "id": point_id(_doc_key(record)),
                "vector": vector,
                "payload": {"url": record.get("url"), "title": record.get("title"),
                            "content": record.get("content") or record.get("text") or ""},
            }
            for record, vector in zip(batch, vectors)
        ])
        if progress:
            progress(min(start + batch_size, len(records)), len(records))

    store.swap_alias(alias, name)  # last: readers only ever see a fully-built index
    return {"collection": name, "alias": alias, "docs": len(records),
            "dim": settings.embed_dim, "model": settings.active_model, "skipped": False}


async def search(
    settings: Settings, embedder: Embedder, store: QdrantStore, query: str, top_k: int = 10
) -> dict:
    """Embed one query (instruction-formatted, same as training) and rank the live index.

    The store is sync httpx; its calls hop to a thread so this can run on the API's
    event loop without blocking it.
    """
    alias = live_alias(settings.qdrant_collection)
    target = await asyncio.to_thread(store.alias_target, alias)
    if target is None:
        raise VectorStoreError(f"서빙 인덱스({alias})가 없습니다 — `uv run rag-index`로 먼저 색인하세요")
    info = await asyncio.to_thread(store.collection_info, target)
    if info and info["dim"] != settings.embed_dim:
        raise VectorStoreError(
            f"인덱스 차원({info['dim']})과 임베더 차원({settings.embed_dim})이 다릅니다 — "
            f"현재 모델({settings.active_model})로 재색인하거나 EMBED_DIM을 맞추세요"
        )

    vector = (await embedder.embed_queries([query]))[0]
    hits = await asyncio.to_thread(store.query, alias, vector, top_k)
    return {"query": query, "collection": target, "model": settings.active_model,
            "hits": [{"score": h["score"], **h["payload"]} for h in hits]}


def index_status(settings: Settings, store: QdrantStore) -> dict:
    """What the serving index looks like right now (drives the UI/status checks)."""
    if not store.ping():
        return {"reachable": False, "alias": live_alias(settings.qdrant_collection),
                "collection": None, "points": 0, "dim": None, "dim_matches": None,
                "collections": []}
    prefix = settings.qdrant_collection
    alias = live_alias(prefix)
    target = store.alias_target(alias)
    info = store.collection_info(target) if target else None
    return {
        "reachable": True,
        "alias": alias,
        "collection": target,
        "points": info["points"] if info else 0,
        "dim": info["dim"] if info else None,
        "dim_matches": (info["dim"] == settings.embed_dim) if info else None,
        "collections": [c for c in store.list_collections() if c.startswith(f"{prefix}__")],
    }


def prune_collections(settings: Settings, store: QdrantStore) -> list[str]:
    """Drop every collection in the family except the alias target (rollback copies
    are kept only until you decide the new index is good — this is that decision)."""
    prefix = settings.qdrant_collection
    live = store.alias_target(live_alias(prefix))
    stale = [c for c in store.list_collections()
             if c.startswith(f"{prefix}__") and c != live]
    for name in stale:
        store.delete_collection(name)
    return stale
