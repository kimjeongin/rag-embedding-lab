"""Synthetic data generation: bootstrap a labeled dataset from an unlabeled corpus.

For each document an Ollama LLM writes the search queries it would answer →
(query, positive) pairs. With hard-negative mining on, the configured embedder finds
the most-similar wrong doc per query and stores it as a hard negative (sharper
contrastive training). The CLI entrypoint (``rag-gen-synthetic``) writes the result.
"""
from __future__ import annotations

import random
import re

import httpx

from rag.config import Settings
from rag.core.entities import Document
from rag.dataset import load_jsonl
from rag.embeddings import build_embedder

_NUMBERING = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")


def _clean_query(line: str) -> str:
    """Strip list numbering/bullets and surrounding quotes from an LLM output line."""
    line = _NUMBERING.sub("", line.strip())
    return line.strip().strip('"').strip("'").strip()


async def _generate_queries(
    http: httpx.AsyncClient, ollama_url: str, model: str, title: str, content: str, n: int
) -> list[str]:
    prompt = (
        f"Write {n} short, natural web-search queries that the document below directly "
        f"answers. Output ONLY the queries, one per line — no numbering, no quotes.\n\n"
        f"Title: {title}\nContent: {content}"
    )
    resp = await http.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        },
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    queries = [q for q in (_clean_query(line) for line in text.splitlines()) if len(q) > 3]
    return queries[:n]


async def _mine_hard_negatives(
    pairs: list[dict], docs: list[dict], settings: Settings, n_negatives: int
) -> None:
    """Attach the top-`n_negatives` most-similar docs that aren't the positive (by the
    configured embedder) to each pair as hard negatives."""
    import numpy as np

    async with build_embedder(settings) as embedder:
        corpus = np.asarray(
            await embedder.embed_documents(
                [Document(content=d["content"], title=d.get("title")) for d in docs]
            ),
            dtype="float32",
        )
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12

        for pair in pairs:
            qvec = np.asarray(await embedder.embed_query(pair["query"]), dtype="float32")
            qvec /= np.linalg.norm(qvec) + 1e-12
            order = np.argsort(-(corpus @ qvec))
            negatives = [i for i in order.tolist() if i != pair["_doc"]][:n_negatives]
            pair["negatives"] = [
                {"title": docs[i].get("title"), "content": docs[i]["content"]} for i in negatives
            ]


async def generate(
    corpus_file: str,
    gen_model: str,
    n_queries: int,
    hard_negatives: int,
    settings: Settings,
    test_fraction: float = 0.25,
    seed: int = 13,
) -> tuple[list[dict], list[dict]]:
    """Generate (train, test) records from the corpus."""
    docs = list(load_jsonl(corpus_file))

    pairs: list[dict] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
        for idx, doc in enumerate(docs):
            queries = await _generate_queries(
                http, settings.ollama_url, gen_model, doc.get("title"), doc["content"], n_queries
            )
            for query in queries:
                pairs.append(
                    {
                        "query": query,
                        "positive": {"title": doc.get("title"), "content": doc["content"]},
                        "_doc": idx,
                    }
                )

    if hard_negatives > 0:
        await _mine_hard_negatives(pairs, docs, settings, hard_negatives)

    for pair in pairs:
        pair.pop("_doc", None)  # internal bookkeeping, not part of the schema

    random.Random(seed).shuffle(pairs)
    n_test = max(1, round(len(pairs) * test_fraction))
    return pairs[n_test:], pairs[:n_test]
