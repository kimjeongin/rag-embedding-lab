"""Offline retrieval evaluation — how well the configured embedder retrieves.

It embeds the WHOLE eval corpus (the "haystack") plus every query with the SAME
Embedder used in serving — so the text is formatted identically (train/inference
parity) — ranks each query's docs by cosine, and scores the ranking against the
qrels with `metrics`.

Ranking is in-memory (numpy), and with the Ollama backend no torch is loaded. This
scales comfortably to tens of thousands of docs; for much larger corpora prefer the
local sentence-transformers backend (batched on-device) or an ANN index.

The corpus must be large and realistic — gold docs PLUS many distractors. If it
contains only the answer docs, every metric saturates near 1.0 and can't tell models
apart; that defeats the point of measuring. See docs/evaluation.md.

Compare models by re-running with a different backend (same corpus, same metrics):
    EMBEDDER=ollama                uv run rag-eval      # baseline
    EMBEDDER=sentence-transformers uv run rag-eval      # a fine-tuned model
"""
from __future__ import annotations

from rag.config import Settings
from rag.core.entities import Document
from rag.core.ports import Embedder
from rag.evaluation.beir import load_corpus, load_qrels, load_queries
from rag.evaluation.metrics import MRR_K, NDCG_K, RECALL_KS, evaluate_rankings

# Only the top results feed our cutoffs, so we keep just this many per query — bounds
# memory on a big corpus without affecting any reported metric.
_TOP_N = max(*RECALL_KS, NDCG_K, MRR_K)


def _top_indices(scores, n: int) -> list[int]:
    """Indices of the n highest scores, best first (argpartition keeps it O(N) on
    large corpora instead of a full sort)."""
    import numpy as np

    if n >= len(scores):
        return np.argsort(-scores).tolist()
    top = np.argpartition(-scores, n)[:n]
    return top[np.argsort(-scores[top])].tolist()


async def rank_corpus(
    embedder: Embedder,
    corpus: dict[str, dict[str, str | None]],
    queries: dict[str, str],
) -> dict[str, list[str]]:
    """Embed the corpus + queries and rank: {query_id: [doc_id, ...]} (top-N, best first).

    Takes an Embedder (not Settings) so it can be unit-tested with an in-memory fake.
    """
    import numpy as np

    doc_ids = list(corpus)
    docs = [Document(content=corpus[d]["text"] or "", title=corpus[d]["title"]) for d in doc_ids]

    matrix = np.asarray(await embedder.embed_documents(docs), dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12

    rankings: dict[str, list[str]] = {}
    for query_id, text in queries.items():
        qvec = np.asarray(await embedder.embed_query(text), dtype="float32")
        qvec /= np.linalg.norm(qvec) + 1e-12
        scores = matrix @ qvec  # cosine — both sides L2-normalised
        rankings[query_id] = [doc_ids[i] for i in _top_indices(scores, _TOP_N)]
    return rankings


async def evaluate(settings: Settings, eval_dir: str) -> dict[str, float]:
    """Load the BEIR-format eval set, rank it with the configured embedder, score it."""
    from rag.embeddings import build_embedder

    corpus = load_corpus(eval_dir)
    queries = load_queries(eval_dir)
    qrels = load_qrels(eval_dir)
    judged = {q: queries[q] for q in queries if qrels.get(q)}  # only scorable queries

    async with build_embedder(settings) as embedder:
        rankings = await rank_corpus(embedder, corpus, judged)
    return evaluate_rankings(rankings, qrels)
