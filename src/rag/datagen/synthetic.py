"""Synthetic data generation: bootstrap a labeled dataset from an unlabeled corpus.

For each document an Ollama LLM writes the **search-box queries** that would lead a
user to it (same language as the document, keyword/question/navigational mix) →
(query, positive) pairs. Two quality gates follow, both running on ONE shared
query×doc similarity pass by the configured embedder:

  - **Round-trip filter** (Promptagator-style consistency), TRAIN SPLIT ONLY: a
    pair survives only if its query actually retrieves its own source doc into the
    top-k. A query that doesn't is noise — too generic, or better answered by
    another page — and the GPL/InPars/Promptagator line of work consistently finds
    filtering beats volume. The held-out test split is deliberately NOT filtered:
    filtering it with the same embedder that evaluation uses would keep only the
    queries that model already gets right, saturating every metric at 1.0 and
    leaving the fine-tune nowhere to go but down.
  - **Hard-negative mining with a false-negative guard** (NV-Retriever
    "TopK-PercPos"): the most-similar wrong docs become hard negatives, but any
    candidate scoring above (1 − margin) × the positive's own score is presumed to
    be a true answer wearing the wrong label and is skipped. RocketQA measured ~70%
    of unfiltered top candidates to be exactly that, and training against them
    teaches the model to push correct answers away.

The CLI entrypoint (``rag-gen-synthetic``) and the SSE route both drive
``generate_stream`` (progress as it goes); ``generate`` collects it.

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
_GEN_RETRIES = 2      # per-doc retries — over hundreds of LLM calls a transient Ollama
                      # hiccup (model swap, load spike) is expected and must not kill the run


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
    # Search-box realism over prose: real site-search queries are short and typed in
    # the site's language — not the polished English questions an LLM defaults to.
    prompt = (
        f"You are looking at one page of a website. Write {n} realistic queries a user "
        f"would type into that site's SEARCH BOX to find this exact page.\n"
        f"Rules:\n"
        f"- Each query must be in the same language as the page (Korean page → Korean queries).\n"
        f"- Mix the styles: mostly short keyword queries (2-4 words), at least one "
        f"natural-language question, and one navigational query (a user hunting for this "
        f"specific page or notice by name).\n"
        f"- Every query must be answerable by THIS page alone.\n"
        f"- Output ONLY the queries, one per line — no numbering, no quotes, no explanations.\n\n"
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


async def _similarity_matrix(pairs: list[dict], docs: list[dict], settings: Settings):
    """The (n_pairs × n_docs) cosine matrix by the configured embedder — ONE embed pass
    shared by the round-trip filter and hard-negative mining. Embedding goes through
    the same adapter as serving, so queries/docs get the same formatting (and the
    adapter owns request batching)."""
    import numpy as np

    from rag.evaluation.retrieval import l2_normalize

    async with build_embedder(settings) as embedder:
        corpus = l2_normalize(np.asarray(
            await embedder.embed_documents(
                [Document(content=d["content"], title=d.get("title")) for d in docs]
            ),
            dtype="float32",
        ))
        qmatrix = l2_normalize(np.asarray(
            await embedder.embed_queries([pair["query"] for pair in pairs]), dtype="float32"
        ))
    return qmatrix @ corpus.T  # (n_pairs, n_docs)


def _round_trip_keep(sims, doc_indices: list[int], k: int) -> list[int]:
    """Indices of the pairs whose own doc ranks in their query's top-k (the
    Promptagator consistency filter — see the module docstring)."""
    import numpy as np

    keep = []
    for i, doc_idx in enumerate(doc_indices):
        top = np.argsort(-sims[i])[:k]
        if doc_idx in top:
            keep.append(i)
    return keep


def _attach_negatives(
    pairs: list[dict], sims, docs: list[dict], n_negatives: int, margin: float
) -> None:
    """Attach up to `n_negatives` most-similar wrong docs per pair as hard negatives —
    skipping probable FALSE negatives: candidates scoring above (1 − margin) × the
    positive's own score (see the module docstring). A pair near-duplicating its doc's
    whole neighborhood may end up with fewer (or zero) negatives; that's the guard
    working, not a bug."""
    import numpy as np

    for pair, row in zip(pairs, sims):
        positive = row[pair["_doc"]]
        ceiling = positive * (1 - margin) if positive > 0 else positive
        order = np.argsort(-row).tolist()
        chosen = [i for i in order if i != pair["_doc"] and row[i] <= ceiling][:n_negatives]
        pair["negatives"] = [
            {"title": docs[i].get("title"), "content": docs[i]["content"]} for i in chosen
        ]


async def generate_stream(
    corpus_file: str,
    gen_model: str,
    n_queries: int,
    hard_negatives: int,
    settings: Settings,
    test_fraction: float = 0.25,
    seed: int = 13,
    round_trip_k: int = 1,
    neg_margin: float = 0.05,
) -> AsyncIterator[dict]:
    """Generate (train, test) records, yielding progress events as it goes.

    Events: ``start`` (doc count + whether thinking was disabled), one ``doc`` per
    finished document (with its queries), ``mining`` before the embed pass,
    ``filtered`` after the round-trip filter (train-split kept/dropped), and a
    terminal ``done`` carrying ``train``/``test`` lists. Per-doc generation is
    bounded-concurrent; pairs are assembled in deterministic doc order before the
    seeded split.

    ``round_trip_k=0`` disables the consistency filter; ``neg_margin=0`` disables
    the false-negative guard (plain top-k mining, the old behaviour).
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
                for attempt in range(_GEN_RETRIES + 1):
                    try:
                        qs = await _generate_queries(
                            http, settings.ollama_url, gen_model,
                            doc.get("title"), doc["content"], n_queries, think,
                        )
                        return idx, doc, qs
                    except httpx.HTTPError:
                        if attempt == _GEN_RETRIES:
                            return idx, doc, []  # this doc contributes nothing; the run survives
                        await asyncio.sleep(2.0 * (attempt + 1))
            return idx, doc, []

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

    # Split FIRST (whole docs held out — their queries are near-paraphrases, see
    # _split_by_doc), so the quality gates can treat the two sides differently:
    # train gets the round-trip filter, test stays unfiltered (module docstring).
    train, test = _split_by_doc(pairs, test_fraction, seed)

    if pairs and (round_trip_k > 0 or hard_negatives > 0):
        yield {"event": "mining", "pairs": len(pairs), "k": hard_negatives}
        sims = await _similarity_matrix(pairs, docs, settings)
        for row, pair in enumerate(pairs):
            pair["_row"] = row  # this pair's row in `sims` — survives the split lists

        if round_trip_k > 0 and train:
            keep = _round_trip_keep(
                sims[[pair["_row"] for pair in train]],
                [pair["_doc"] for pair in train],
                round_trip_k,
            )
            yield {
                "event": "filtered",
                "kept": len(keep),
                "dropped": len(train) - len(keep),
                "k": round_trip_k,
            }
            train = [train[i] for i in keep]
        if hard_negatives > 0:
            survivors = train + test
            _attach_negatives(
                survivors, sims[[pair["_row"] for pair in survivors]],
                docs, hard_negatives, neg_margin,
            )

    for pair in (*train, *test):
        pair.pop("_doc", None)   # internal bookkeeping, not part of the schema
        pair.pop("_row", None)
    yield {"event": "done", "train": train, "test": test}


async def generate(
    corpus_file: str,
    gen_model: str,
    n_queries: int,
    hard_negatives: int,
    settings: Settings,
    test_fraction: float = 0.25,
    seed: int = 13,
    round_trip_k: int = 1,
    neg_margin: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """Collect ``generate_stream`` into (train, test) — for the CLI / non-streaming callers."""
    train: list[dict] = []
    test: list[dict] = []
    async for event in generate_stream(
        corpus_file, gen_model, n_queries, hard_negatives, settings,
        test_fraction, seed, round_trip_k, neg_margin,
    ):
        if event["event"] == "done":
            train, test = event["train"], event["test"]
    return train, test
