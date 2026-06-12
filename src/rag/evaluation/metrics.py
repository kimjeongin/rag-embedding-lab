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
import random
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


def per_query_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    recall_ks: Sequence[int] = RECALL_KS,
) -> dict[str, dict[str, float]]:
    """``{query_id: {metric: score}}`` for every judged query.

    The raw scores behind ``evaluate_rankings``' averages — keep them: they are what
    confidence intervals and paired run-vs-run comparisons need. Queries without
    qrels are skipped (you can't score them). ``recall_ks`` lets the caller extend the
    cutoffs — e.g. recall@50 when the model's production job is candidate generation
    for a reranker (the ranking must be at least that deep).
    """
    out: dict[str, dict[str, float]] = {}
    for query_id, ranked in rankings.items():
        judgments = qrels.get(query_id)
        if not judgments:
            continue
        relevant = set(judgments)
        row = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in recall_ks}
        row[f"mrr@{MRR_K}"] = reciprocal_rank(ranked, relevant, MRR_K)
        row[f"ndcg@{NDCG_K}"] = ndcg_at_k(ranked, judgments, NDCG_K)
        out[query_id] = row
    return out


def mean_metrics(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Per-metric mean over the per-query scores ({} when nothing was judged).

    Keys come from the per-query rows (insertion-ordered): recall@k…, mrr@10, ndcg@10.
    """
    n = len(per_query)
    if n == 0:
        return {}
    keys = list(next(iter(per_query.values())))
    return {key: sum(row[key] for row in per_query.values()) / n for key in keys}


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    recall_ks: Sequence[int] = RECALL_KS,
) -> dict[str, float]:
    """Average recall@k / MRR@10 / nDCG@10 over every query that has judgments.

    Queries without qrels are skipped (you can't score them). Returns an
    insertion-ordered dict: recall@1, recall@3, recall@5, recall@10, …, mrr@10, ndcg@10.
    """
    return mean_metrics(per_query_metrics(rankings, qrels, recall_ks))


def bootstrap_ci(
    per_query: Mapping[str, Mapping[str, float]],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile-bootstrap confidence interval of each metric's mean: {metric: (lo, hi)}.

    Resamples queries with replacement — on a deterministic full-corpus ranking the
    ONLY noise in an average is which queries the eval set happens to contain, so
    this is exactly the uncertainty a small set needs quantified (on ~50 queries one
    query swings recall@1 by ~2 points). Seeded → reproducible.
    """
    rows = list(per_query.values())
    n = len(rows)
    if n == 0:
        return {}
    keys = list(rows[0])
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(n_resamples):
        resample = rng.choices(rows, k=n)
        for key in keys:
            samples[key].append(sum(row[key] for row in resample) / n)
    alpha = (1.0 - confidence) / 2.0
    out: dict[str, tuple[float, float]] = {}
    for key, values in samples.items():
        values.sort()
        out[key] = (_quantile(values, alpha), _quantile(values, 1.0 - alpha))
    return out


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list (numpy's default method)."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
