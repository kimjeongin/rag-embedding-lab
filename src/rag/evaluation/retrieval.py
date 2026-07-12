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

To attribute a fine-tune's effect, measure base and fine-tuned with the SAME backend —
quantisation/pooling/truncation differ between stacks and would pollute the Δ:
    ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval  # base
    ST_MODEL=outputs/embedding-ft      uv run rag-eval  # tuned
(An EMBEDDER=ollama run is a parity check against an Ollama-served stack, not the baseline.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from rag.config import Settings
from rag.core.entities import Document
from rag.core.ports import Embedder
from rag.evaluation.beir import load_corpus, load_qrels, load_queries, resolve_split
from rag.evaluation.metrics import (
    MRR_K,
    NDCG_K,
    RECALL_KS,
    bootstrap_ci,
    mean_metrics,
    per_query_metrics,
)

# Default ranking depth. In a hybrid + reranker pipeline the dense model's job is
# CANDIDATE GENERATION — what matters is whether the answer is in the top-K handed to
# fusion/reranking, not its exact rank. Set EVAL_TOP_K to your production fusion
# depth so recall@K measures the model's actual job.
DEFAULT_TOP_K = 50


def eval_top_k() -> int:
    """Ranking depth (EVAL_TOP_K, default 50) — never below the fixed metric cutoffs."""
    try:
        requested = int(os.getenv("EVAL_TOP_K", str(DEFAULT_TOP_K)))
    except ValueError:
        requested = DEFAULT_TOP_K
    return max(requested, *RECALL_KS, NDCG_K, MRR_K)


def _top_indices(scores, n: int) -> list[int]:
    """Indices of the n highest scores, best first (argpartition keeps it O(N) on
    large corpora instead of a full sort)."""
    import numpy as np

    if n >= len(scores):
        return np.argsort(-scores).tolist()
    top = np.argpartition(-scores, n)[:n]
    return top[np.argsort(-scores[top])].tolist()


def l2_normalize(matrix):
    """Row-wise L2 normalise so a dot product equals cosine similarity — shared with
    the datagen mining pass (rag.datagen.synthetic), which scores the same way."""
    import numpy as np

    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


async def rank_corpus(
    embedder: Embedder,
    corpus: dict[str, dict[str, str | None]],
    queries: dict[str, str],
    top_n: int | None = None,
) -> dict[str, list[str]]:
    """Embed the corpus + queries and rank: {query_id: [doc_id, ...]} (top-N, best first).

    Documents and queries are each embedded in a single batch call, then scored as one
    (queries × docs) matrix product — no per-query round-trip. Takes an Embedder (not
    Settings) so it can be unit-tested with an in-memory fake. ``top_n`` bounds memory
    on a big corpus without affecting any metric at or below that depth.
    """
    import numpy as np

    top_n = top_n or eval_top_k()
    doc_ids = list(corpus)
    query_ids = list(queries)
    if not doc_ids or not query_ids:
        return {q: [] for q in query_ids}

    docs = [Document(content=corpus[d]["text"] or "", title=corpus[d]["title"]) for d in doc_ids]
    doc_matrix = l2_normalize(np.asarray(await embedder.embed_documents(docs), dtype="float32"))
    query_matrix = l2_normalize(
        np.asarray(await embedder.embed_queries([queries[q] for q in query_ids]), dtype="float32")
    )

    sims = query_matrix @ doc_matrix.T  # (n_queries, n_docs) cosine — both sides normalised
    return {
        query_id: [doc_ids[i] for i in _top_indices(row, top_n)]
        for query_id, row in zip(query_ids, sims)
    }


@dataclass(frozen=True)
class EvalReport:
    """One model's scores on one eval set.

    ``metrics`` are the headline averages; ``per_query`` holds the raw scores they
    average (what CIs and paired run comparisons need); ``ci95`` is the bootstrap
    95% interval of each average — without it, a Δ between two runs on a small set
    can't be told apart from query-sampling noise. ``rankings`` keeps each query's
    top results so a run-vs-run diff can show WHAT was retrieved, not just scores.
    """

    metrics: dict[str, float]                   # {metric: mean} ({} if nothing judged)
    per_query: dict[str, dict[str, float]]      # {query_id: {metric: score}}
    ci95: dict[str, tuple[float, float]]        # {metric: (lo, hi)}
    rankings: dict[str, list[str]] = field(default_factory=dict)  # {query_id: [doc_id, …]}
    split: str = "test"                          # which qrels split scored this


def metric_recall_ks(top_k: int) -> tuple[int, ...]:
    """The recall cutoffs to report for a ranking of this depth."""
    if top_k > max(RECALL_KS):
        return (*RECALL_KS, top_k)
    return RECALL_KS


async def evaluate(settings: Settings, eval_dir: str, split: str = "dev") -> EvalReport:
    """Load the BEIR-format eval set, rank it with the configured embedder, score it."""
    from rag.embeddings import build_embedder

    resolved = resolve_split(eval_dir, split)
    corpus = load_corpus(eval_dir)
    queries = load_queries(eval_dir)
    qrels = load_qrels(eval_dir, resolved)
    judged = {q: queries[q] for q in queries if qrels.get(q)}  # only scorable queries

    top_k = eval_top_k()
    async with build_embedder(settings) as embedder:
        rankings = await rank_corpus(embedder, corpus, judged, top_n=top_k)

    per_query = per_query_metrics(rankings, qrels, metric_recall_ks(top_k))
    return EvalReport(
        metrics=mean_metrics(per_query),
        per_query=per_query,
        ci95=bootstrap_ci(per_query),
        rankings={q: ranked[:10] for q, ranked in rankings.items()},  # top-10 is plenty for diffs
        split=resolved,
    )
