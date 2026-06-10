"""rag-eval metric definitions — pure, no embedder/numpy needed."""
import math

from rag.evaluation.metrics import (
    bootstrap_ci,
    evaluate_rankings,
    mean_metrics,
    ndcg_at_k,
    per_query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_in_top_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a"}, 1) == 1.0
    assert recall_at_k(ranked, {"c"}, 1) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    # two relevant docs, only one in the top-2 → 0.5
    assert recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_reciprocal_rank_first_hit():
    ranked = ["x", "y", "z"]
    assert reciprocal_rank(ranked, {"x"}, 10) == 1.0
    assert reciprocal_rank(ranked, {"y"}, 10) == 0.5
    assert reciprocal_rank(ranked, {"w"}, 10) == 0.0
    # relevant doc sits outside the cutoff
    assert reciprocal_rank(["a", "b", "c"], {"c"}, 2) == 0.0


def test_ndcg_binary_and_graded():
    # one relevant doc at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert abs(ndcg_at_k(["a", "b", "c"], {"b": 1.0}, 10) - (1 / math.log2(3))) < 1e-9
    # a perfect ranking scores 1.0
    assert abs(ndcg_at_k(["b", "a"], {"b": 1.0}, 10) - 1.0) < 1e-9
    # graded gains: ideal ranking puts the gain-3 doc first
    gains = {"a": 3.0, "c": 1.0}
    actual = 3.0 / math.log2(2) + 1.0 / math.log2(4)  # a@1, c@3
    ideal = 3.0 / math.log2(2) + 1.0 / math.log2(3)   # a@1, c@2
    assert abs(ndcg_at_k(["a", "b", "c"], gains, 10) - actual / ideal) < 1e-9
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_evaluate_rankings_aggregates_and_skips_unjudged():
    rankings = {
        "q1": ["d1", "d2", "d3"],   # relevant d1 at rank 1
        "q2": ["d9", "d2", "d3"],   # relevant d2 at rank 2
        "q3": ["d1", "d2"],         # no qrels → skipped entirely
    }
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    m = evaluate_rankings(rankings, qrels)
    assert m["recall@1"] == 0.5          # q1 hit, q2 missed at rank 1
    assert m["recall@3"] == 1.0
    assert abs(m["mrr@10"] - (1.0 + 0.5) / 2) < 1e-9
    # insertion order is stable and report-friendly
    assert list(m) == ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def test_evaluate_rankings_empty_when_no_judged_queries():
    assert evaluate_rankings({"q1": ["d1"]}, {}) == {}


def test_per_query_metrics_returns_raw_scores():
    rankings = {
        "q1": ["d1", "d2"],   # hit at rank 1
        "q2": ["d9", "d2"],   # hit at rank 2
        "q3": ["d1"],         # no qrels → skipped
    }
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    per_query = per_query_metrics(rankings, qrels)

    assert set(per_query) == {"q1", "q2"}
    assert per_query["q1"]["recall@1"] == 1.0
    assert per_query["q2"]["recall@1"] == 0.0
    assert per_query["q2"]["mrr@10"] == 0.5
    # the headline averages are exactly the means of these rows
    assert evaluate_rankings(rankings, qrels) == mean_metrics(per_query)


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    # 40 queries, half scoring 1.0 and half 0.0 → mean 0.5 with real spread
    per_query = {f"q{i}": {"ndcg@10": 1.0 if i % 2 else 0.0} for i in range(40)}
    ci = bootstrap_ci(per_query)
    lo, hi = ci["ndcg@10"]

    assert lo <= 0.5 <= hi               # the interval brackets the observed mean
    assert 0.0 < lo < hi < 1.0           # and is non-degenerate on a mixed sample
    assert bootstrap_ci(per_query) == ci  # seeded → reproducible
    assert bootstrap_ci({}) == {}
