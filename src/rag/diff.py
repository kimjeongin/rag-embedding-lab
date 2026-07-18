"""Run-vs-run comparison — per-query win/loss, paired significance, topic slices.

Two aggregate scores on a small query set can't be told apart from sampling noise by
eye, and overlapping confidence intervals are the WRONG test (each run's CI ignores
that both runs scored the *same* queries). The honest comparison is paired: take each
query's score under A and under B, look at the per-query deltas, and ask how often a
mean delta this large appears if the sign of each delta were random — a sign-flip
permutation test. "B wins 32 / loses 9 / ties 7, p=0.003" ends arguments that
"0.91 vs 0.93" starts.

Pure stdlib (same stance as rag.evaluation.metrics) — unit-testable, no API/ML stack.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

DEFAULT_METRIC = "ndcg@10"


def paired_permutation_test(deltas: Sequence[float], n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for "mean(delta) ≠ 0".

    Under the null (the two runs are equivalent) each query's delta is as likely to
    be +d as −d, so flipping signs at random generates the null distribution of the
    mean delta. The p-value is the fraction of flips at least as extreme as what we
    observed (+1 smoothing keeps it strictly positive). Seeded → reproducible.
    """
    n = len(deltas)
    if n == 0 or all(d == 0 for d in deltas):
        return 1.0
    observed = abs(sum(deltas) / n)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in deltas) / n
        if abs(flipped) >= observed - 1e-12:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_resamples + 1)


def _topic_of(query_id: str) -> str | None:
    """The topic key inside ids like ``q-3-1`` → "3" (None when the id has no topic)."""
    parts = query_id.split("-")
    return parts[1] if len(parts) >= 3 and parts[0] == "q" else None


def candidate_union(
    run_a: Mapping,
    run_b: Mapping,
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
    slice_map: Mapping[str, str] | None = None,
) -> dict | None:
    """후보군 상보성 — recall@k of A alone, B alone, and A∪B (B relative to A).

    하이브리드 검색에서 dense의 실제 가치는 단독 점수가 아니라 **다른 랭커(BM25)가
    놓친 정답을 후보군에 보태는 양**이다: ``marginal_b = recall(A∪B) − recall(A)``.
    두 랭커의 top-k를 합친 후보 집합으로 relevant 문서를 얼마나 커버하는지 재므로,
    "뒤에 리랭커가 있을 때 B를 추가하면 천장이 얼마나 올라가나"에 답한다.

    Both runs must share an eval fingerprint (else ValueError). Stored rankings are
    top-10 deep, so k > 10 is clamped by the data itself. Returns None when either
    run has no stored rankings (legacy records).
    """
    fp_a, fp_b = run_a.get("eval_fingerprint"), run_b.get("eval_fingerprint")
    if not fp_a or fp_a != fp_b:
        raise ValueError("두 런의 평가셋이 다릅니다 — 같은 평가셋(fingerprint)에서 측정된 런끼리만 비교할 수 있어요")
    rankings_a, rankings_b = run_a.get("rankings") or {}, run_b.get("rankings") or {}
    if not rankings_a or not rankings_b:
        return None
    common = [q for q in rankings_a if q in rankings_b and qrels.get(q)]
    if not common:
        return None

    def _recall(found: set[str], relevant: set[str]) -> float:
        return len(found & relevant) / len(relevant)

    rows = []
    for query_id in common:
        relevant = {d for d, gain in qrels[query_id].items() if gain > 0}
        top_a, top_b = set(rankings_a[query_id][:k]), set(rankings_b[query_id][:k])
        rows.append((
            query_id,
            _recall(top_a, relevant),
            _recall(top_b, relevant),
            _recall(top_a | top_b, relevant),
        ))

    def _mean(subset: list[tuple[str, float, float, float]]) -> dict:
        n = len(subset)
        recall_a = sum(r[1] for r in subset) / n
        recall_b = sum(r[2] for r in subset) / n
        union = sum(r[3] for r in subset) / n
        return {
            "n": n,
            "recall_a": recall_a,
            "recall_b": recall_b,
            "recall_union": union,
            "marginal_b": union - recall_a,   # B가 A의 후보군 위에 보태는 정답
            "marginal_a": union - recall_b,
        }

    slices: list[dict] = []
    if slice_map:
        by_name: dict[str, list] = {}
        for row in rows:
            name = slice_map.get(row[0])
            if name:
                by_name.setdefault(name, []).append(row)
        if len(by_name) >= 2:
            slices = [{"topic": name, **_mean(group)} for name, group in sorted(by_name.items())]

    return {"k": k, **_mean(rows), "slices": slices}


def compare_runs(
    run_a: Mapping,
    run_b: Mapping,
    metric: str = DEFAULT_METRIC,
    slice_map: Mapping[str, str] | None = None,
) -> dict:
    """Paired comparison of two registry records (B relative to A; +delta = B better).

    Both runs must carry per-query scores and share an eval fingerprint — scores from
    different eval-set contents (or splits) are not comparable and raise ValueError.
    ``slice_map`` ({query_id: slice}) groups the slice table by the eval set's own
    query tags (e.g. standard/jargon); without it the id shape (``q-3-1`` → "3") is
    the fallback.
    """
    fp_a, fp_b = run_a.get("eval_fingerprint"), run_b.get("eval_fingerprint")
    if not fp_a or fp_a != fp_b:
        raise ValueError("두 런의 평가셋이 다릅니다 — 같은 평가셋(fingerprint)에서 측정된 런끼리만 비교할 수 있어요")
    pq_a, pq_b = run_a.get("per_query") or {}, run_b.get("per_query") or {}
    common = [q for q in pq_a if q in pq_b and metric in pq_a[q] and metric in pq_b[q]]
    if not common:
        raise ValueError("두 런에 공통 per-query 점수가 없습니다 (예전 기록이거나 지표가 다릅니다)")

    queries = []
    for query_id in common:
        a, b = pq_a[query_id][metric], pq_b[query_id][metric]
        queries.append({"query_id": query_id, "a": a, "b": b, "delta": b - a})
    queries.sort(key=lambda q: q["delta"])  # worst regressions first — where to look

    deltas = [q["delta"] for q in queries]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    mean_a = sum(q["a"] for q in queries) / len(queries)
    mean_b = sum(q["b"] for q in queries) / len(queries)

    # Every metric both runs share gets its own paired test — a model can win nDCG
    # while losing recall@50, and that disagreement is worth surfacing.
    shared_metrics = [m for m in (pq_a[common[0]] or {}) if m in pq_b[common[0]]]
    by_metric = {}
    for m in shared_metrics:
        m_deltas = [pq_b[q][m] - pq_a[q][m] for q in common]
        by_metric[m] = {
            "mean_a": sum(pq_a[q][m] for q in common) / len(common),
            "mean_b": sum(pq_b[q][m] for q in common) / len(common),
            "delta": sum(m_deltas) / len(common),
            "p_value": paired_permutation_test(m_deltas),
        }

    # Topic slices — an average can rise while one topic quietly breaks.
    slices: list[dict] = []
    by_topic: dict[str, list[str]] = {}
    for query_id in common:
        topic = slice_map.get(query_id) if slice_map else _topic_of(query_id)
        if topic is not None:
            by_topic.setdefault(topic, []).append(query_id)
    if len(by_topic) >= 2:
        for topic in sorted(by_topic, key=lambda t: (len(t), t)):
            ids = by_topic[topic]
            t_mean_a = sum(pq_a[q][metric] for q in ids) / len(ids)
            t_mean_b = sum(pq_b[q][metric] for q in ids) / len(ids)
            slices.append({
                "topic": topic,
                "n": len(ids),
                "mean_a": t_mean_a,
                "mean_b": t_mean_b,
                "delta": t_mean_b - t_mean_a,
            })

    return {
        "metric": metric,
        "n": len(queries),
        "wins": wins,
        "losses": losses,
        "ties": len(queries) - wins - losses,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_b - mean_a,
        "p_value": paired_permutation_test(deltas),
        "queries": queries,
        "by_metric": by_metric,
        "slices": slices,
    }
