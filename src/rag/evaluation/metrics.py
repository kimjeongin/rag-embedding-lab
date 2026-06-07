"""Pure retrieval metrics — recall@k, MRR@k, nDCG@k over ranked doc-id lists.

No numpy, no embedder: this module takes the ranking your retriever already produced
and the relevance judgments (qrels), and returns the aggregate scores. Definitions
follow the BEIR / trec_eval conventions (linear gains for nDCG) so the numbers are
comparable to published results.

    rankings : {query_id: [doc_id, ...]}    # best-first — your retriever's output
    qrels    : {query_id: {doc_id: gain}}   # gain > 0 == relevant (graded gains ok)

Keeping the metrics pure (and separate from the embedding/ranking step) makes them
trivially unit-testable and reusable by any retriever, not just this one.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# Reported cutoffs. recall@{1,3,5,10} shows *where* the relevant docs land; nDCG@10
# is BEIR's headline number (rewards putting them higher); MRR@10 summarises the
# rank of the first hit.
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)
NDCG_K = 10
MRR_K = 10


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant docs that appear in the top-k."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant doc within the top-k (0 if none)."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an ordered gain list (position i discounted by
    1/log2(i+1), i 1-based)."""
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalised DCG@k — DCG of the ranking over DCG of the ideal ordering."""
    if not gains:
        return 0.0
    actual = _dcg([gains.get(doc_id, 0.0) for doc_id in ranked[:k]])
    ideal = _dcg(sorted(gains.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Average recall@k / MRR@10 / nDCG@10 over every query that has judgments.

    Queries without qrels are skipped (you can't score them). Returns an
    insertion-ordered dict: recall@1, recall@3, recall@5, recall@10, mrr@10, ndcg@10.
    """
    query_ids = [q for q in rankings if qrels.get(q)]
    n = len(query_ids)
    if n == 0:
        return {}

    out: dict[str, float] = {}
    for k in RECALL_KS:
        out[f"recall@{k}"] = (
            sum(recall_at_k(rankings[q], set(qrels[q]), k) for q in query_ids) / n
        )
    out[f"mrr@{MRR_K}"] = (
        sum(reciprocal_rank(rankings[q], set(qrels[q]), MRR_K) for q in query_ids) / n
    )
    out[f"ndcg@{NDCG_K}"] = (
        sum(ndcg_at_k(rankings[q], qrels[q], NDCG_K) for q in query_ids) / n
    )
    return out
