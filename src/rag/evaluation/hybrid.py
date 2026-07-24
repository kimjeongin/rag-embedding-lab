"""Hybrid retrieval — fuse the dense ranking with the lexical (BM25) ranking.

프로덕션 검색은 BM25 + dense + 리랭커다. 3.4의 "상보성"은 두 랭커의 후보 **합집합
recall**로 dense가 BM25 위에 보태는 정답을 셌다 — 진단이지 시스템이 아니다. 여기서는
그 두 랭킹을 실제로 하나로 **융합**해, 융합 결과가 각 성분 단독을 이기는지( = 하이브리드가
실제로 좋아지는지)를 같은 지표로 잰다.

융합은 **Reciprocal Rank Fusion**(RRF)이다 — 점수 스케일이 다른 두 랭커(코사인 vs BM25)를
섞는 표준 방법으로, 점수 자체가 아니라 **순위**만 쓰기 때문에 정규화가 필요 없다:

    score(d) = Σ_i  w_i / (k + rank_i(d))          # rank 1-based, 목록에 없으면 0 기여

튜닝 노브는 dense 가중 α 하나다 (lexical 가중 = 1-α). k는 RRF 표준값 60으로 고정한다
(k는 상위 순위 간 상대 간격만 조절하는 둔감한 파라미터라, 작은 dev에서 과적합을 피하려
가중 하나만 움직인다 — Google Tuning Playbook의 nuisance-parameter 취급과 같은 결).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_K = 60


def rrf_fuse(
    components: Sequence[tuple[float, Mapping[str, Sequence[str]]]],
    *,
    k: int = DEFAULT_K,
    depth: int = 50,
) -> dict[str, list[str]]:
    """Weighted Reciprocal Rank Fusion of 2+ rankings → {query_id: [doc_id, ...]}.

    ``components`` is a list of ``(weight, {query_id: best-first doc_ids})``. Every
    query present in ANY component is fused; a doc missing from a component simply
    scores 0 there. Ties break on doc_id for a stable, reproducible order. The output
    is truncated to ``depth`` (enough for the metric cutoffs).
    """
    query_ids: list[str] = []
    seen: set[str] = set()
    for _, ranking in components:
        for qid in ranking:
            if qid not in seen:
                seen.add(qid)
                query_ids.append(qid)

    fused: dict[str, list[str]] = {}
    for qid in query_ids:
        scores: dict[str, float] = {}
        for weight, ranking in components:
            if not weight:
                continue
            for rank, doc_id in enumerate(ranking.get(qid, ()), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
        ordered = sorted(scores, key=lambda d: (-scores[d], d))
        fused[qid] = ordered[:depth]
    return fused


def fuse_dense_lexical(
    dense: Mapping[str, Sequence[str]],
    lexical: Mapping[str, Sequence[str]],
    *,
    alpha: float,
    k: int = DEFAULT_K,
    depth: int = 50,
) -> dict[str, list[str]]:
    """Convenience 2-way fuse: ``alpha`` weights dense, ``1 - alpha`` weights lexical.

    ``alpha=1`` is dense-only, ``alpha=0`` is lexical-only — the sweep endpoints are the
    component baselines themselves, so the fused curve and its baselines share one code
    path (no separate "dense alone" scorer to drift out of parity).
    """
    return rrf_fuse([(alpha, dense), (1.0 - alpha, lexical)], k=k, depth=depth)
