"""Synthetic data generation: bootstrap a labeled dataset from an unlabeled corpus.

For each document an Ollama LLM writes the search queries it would answer →
(query, positive) pairs. With hard-negative mining on, the configured embedder finds
the most-similar wrong doc per query and stores it as a hard negative (sharper
contrastive training). The CLI entrypoint (``rag-gen-synthetic``) and the SSE route
both drive ``generate_stream`` (progress as it goes); ``generate`` collects it.

Two things keep it from *looking* hung (the failure mode that bit us with a reasoning
model): per-doc query generation runs **concurrently** (bounded), and if the model
supports "thinking" we **turn it off** — a reasoning model spends ~80s/doc emitting
thought tokens before the answer, vs ~1.5s with thinking off.
"""
from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator

import httpx

from rag.config import Settings
from rag.core.entities import Document
from rag.dataset import load_jsonl
from rag.embeddings import build_embedder

_NUMBERING = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")
_GEN_CONCURRENCY = 4  # parallel per-doc chat calls (bounded so a laptop GPU isn't swamped)


def _clean_query(line: str) -> str:
    """Strip list numbering/bullets and surrounding quotes from an LLM output line."""
    line = _NUMBERING.sub("", line.strip())
    return line.strip().strip('"').strip("'").strip()


def _normalize_query(query: str) -> str:
    """Lowercased, whitespace-collapsed form — the key for duplicate detection."""
    return " ".join(query.lower().split())


def _dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Drop pairs whose query repeats another (case/whitespace-insensitive), keeping the
    first occurrence and the original order. The LLM often re-phrases the same question
    across documents; near-identical anchors add no contrastive signal."""
    seen: set[str] = set()
    out: list[dict] = []
    for pair in pairs:
        key = _normalize_query(pair["query"])
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out


def _split_by_doc(pairs: list[dict], test_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split into (train, test) by holding out whole DOCUMENTS, not individual pairs.

    A document's generated queries are near-paraphrases of each other; if some landed in
    train and others in test, the test score would be inflated by that overlap. Holding
    out whole docs keeps the split honest. Deterministic given `seed`; a single-doc
    corpus can't be split, so everything stays in train.
    """
    doc_ids = list(dict.fromkeys(pair["_doc"] for pair in pairs))
    shuffled = doc_ids[:]
    random.Random(seed).shuffle(shuffled)
    test_docs = set(shuffled[: round(len(doc_ids) * test_fraction)])
    train = [p for p in pairs if p["_doc"] not in test_docs]
    test = [p for p in pairs if p["_doc"] in test_docs]
    return train, test


async def _supports_thinking(http: httpx.AsyncClient, ollama_url: str, model: str) -> bool:
    """Whether `model` is a reasoning model (so we can disable thinking for speed)."""
    try:
        resp = await http.post(f"{ollama_url}/api/show", json={"model": model})
        resp.raise_for_status()
        return "thinking" in (resp.json().get("capabilities") or [])
    except httpx.HTTPError:
        return False


async def _generate_queries(
    http: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    title: str,
    content: str,
    n: int,
    think: bool | None = None,
) -> list[str]:
    prompt = (
        f"Write {n} short, natural web-search queries that the document below directly "
        f"answers. Output ONLY the queries, one per line — no numbering, no quotes.\n\n"
        f"Title: {title}\nContent: {content}"
    )
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    if think is not None:  # only send the field when we know the model's capability
        payload["think"] = think
    resp = await http.post(f"{ollama_url}/api/chat", json=payload)
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    queries = [q for q in (_clean_query(line) for line in text.splitlines()) if len(q) > 3]
    return queries[:n]


async def _mine_hard_negatives(
    pairs: list[dict], docs: list[dict], settings: Settings, n_negatives: int
) -> None:
    """Attach the top-`n_negatives` most-similar docs that aren't the positive (by the
    configured embedder) to each pair as hard negatives.

    Corpus and all pair queries are embedded in one batch each, then scored as a single
    (queries × docs) matrix product.
    """
    import numpy as np

    def _l2(matrix):
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

    async with build_embedder(settings) as embedder:
        corpus = _l2(np.asarray(
            await embedder.embed_documents(
                [Document(content=d["content"], title=d.get("title")) for d in docs]
            ),
            dtype="float32",
        ))
        qmatrix = _l2(np.asarray(
            await embedder.embed_queries([pair["query"] for pair in pairs]), dtype="float32"
        ))

        sims = qmatrix @ corpus.T  # (n_pairs, n_docs)
        for pair, row in zip(pairs, sims):
            order = np.argsort(-row).tolist()
            negatives = [i for i in order if i != pair["_doc"]][:n_negatives]
            pair["negatives"] = [
                {"title": docs[i].get("title"), "content": docs[i]["content"]} for i in negatives
            ]


async def generate_stream(
    corpus_file: str,
    gen_model: str,
    n_queries: int,
    hard_negatives: int,
    settings: Settings,
    test_fraction: float = 0.25,
    seed: int = 13,
) -> AsyncIterator[dict]:
    """Generate (train, test) records, yielding progress events as it goes.

    Events: ``start`` (doc count + whether thinking was disabled), one ``doc`` per
    finished document (with its queries), ``mining`` before hard-negative mining, and a
    terminal ``done`` carrying ``train``/``test`` lists. Per-doc generation is bounded-
    concurrent; pairs are assembled in deterministic doc order before the seeded split.
    """
    docs = list(load_jsonl(corpus_file))
    if not docs:
        yield {"event": "done", "train": [], "test": []}
        return

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
        # A reasoning model would spend ~80s/doc on thought tokens — turn it off.
        think = False if await _supports_thinking(http, settings.ollama_url, gen_model) else None
        yield {"event": "start", "docs": len(docs), "model": gen_model, "thinking_disabled": think is False}

        sem = asyncio.Semaphore(_GEN_CONCURRENCY)

        async def _one(idx: int, doc: dict) -> tuple[int, dict, list[str]]:
            async with sem:
                qs = await _generate_queries(
                    http, settings.ollama_url, gen_model, doc.get("title"), doc["content"], n_queries, think
                )
            return idx, doc, qs

        tasks = [asyncio.create_task(_one(i, d)) for i, d in enumerate(docs)]
        results: list[list[str]] = [[] for _ in docs]
        completed = 0
        for fut in asyncio.as_completed(tasks):
            idx, doc, qs = await fut
            results[idx] = qs
            completed += 1
            yield {
                "event": "doc",
                "done": completed,
                "total": len(docs),
                "title": doc.get("title"),
                "queries": qs,
            }

    # Assemble in deterministic doc order (independent of completion order above), then
    # drop duplicate queries the LLM repeated across docs.
    pairs: list[dict] = []
    for idx, doc in enumerate(docs):
        for query in results[idx]:
            pairs.append(
                {"query": query, "positive": {"title": doc.get("title"), "content": doc["content"]}, "_doc": idx}
            )
    pairs = _dedup_pairs(pairs)

    if hard_negatives > 0 and pairs:
        yield {"event": "mining", "pairs": len(pairs), "k": hard_negatives}
        await _mine_hard_negatives(pairs, docs, settings, hard_negatives)

    # Hold out whole docs (their queries are near-paraphrases — see _split_by_doc).
    train, test = _split_by_doc(pairs, test_fraction, seed)
    for pair in (*train, *test):
        pair.pop("_doc", None)  # internal bookkeeping, not part of the schema
    yield {"event": "done", "train": train, "test": test}


async def generate(
    corpus_file: str,
    gen_model: str,
    n_queries: int,
    hard_negatives: int,
    settings: Settings,
    test_fraction: float = 0.25,
    seed: int = 13,
) -> tuple[list[dict], list[dict]]:
    """Collect ``generate_stream`` into (train, test) — for the CLI / non-streaming callers."""
    train: list[dict] = []
    test: list[dict] = []
    async for event in generate_stream(
        corpus_file, gen_model, n_queries, hard_negatives, settings, test_fraction, seed
    ):
        if event["event"] == "done":
            train, test = event["train"], event["test"]
    return train, test
