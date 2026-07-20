"""클릭로그 클리닝 규칙 + 노이즈 리허설 — 규칙별 동작과 naive 대비 우위를 고정한다."""
from __future__ import annotations

from rag.datagen.clicklog import CleanConfig, clean, contains_pii, naive_pairs
from rag.datagen.clicklog_sim import SimConfig, simulate


def _event(session: str, query: str, clicks: list[dict], results: list[str] | None = None) -> dict:
    return {"session": session, "query": query, "results": results or [], "clicks": clicks}


def test_pii_queries_are_dropped_entirely():
    assert contains_pii("계정 잠김 kim.cs@daon.example")
    assert contains_pii("연락처 010-1234-5678 로 문의")
    assert not contains_pii("머니핀 정산 방법")

    out = clean([
        _event("s1", "비번 초기화 kim.cs@daon.example",
               [{"doc_id": "d1", "rank": 1, "dwell_sec": 100}]),
    ])
    assert out.pairs == []
    assert out.report["dropped_pii"] == 1


def test_short_dwell_clicks_are_bounces_not_positives():
    out = clean([
        _event("s1", "룸북 예약", [
            {"doc_id": "wrong", "rank": 1, "dwell_sec": 3},
            {"doc_id": "gold", "rank": 2, "dwell_sec": 120},
        ]),
    ])
    assert [(p["query"], p["doc_id"]) for p in out.pairs] == [("룸북 예약", "gold")]
    assert out.report["bounces_ignored"] == 1


def test_reformulation_transfers_failed_query_to_final_doc():
    """은어 쿼리(만족 클릭 없음)가 세션의 최종 만족 문서에 연결된다 — 은어 supervision."""
    out = clean([
        _event("s1", "두꺼비 정산 안돼", [{"doc_id": "top", "rank": 1, "dwell_sec": 4}]),
        _event("s1", "머니핀 정산", [{"doc_id": "gold", "rank": 1, "dwell_sec": 90}]),
    ])
    got = {(p["query"], p["doc_id"]) for p in out.pairs}
    assert ("두꺼비 정산 안돼", "gold") in got          # 전이된 쌍
    assert ("머니핀 정산", "gold") in got               # 직접 쌍
    assert out.report["positives_transferred"] == 1

    off = clean([
        _event("s1", "두꺼비 정산 안돼", []),
        _event("s1", "머니핀 정산", [{"doc_id": "gold", "rank": 1, "dwell_sec": 90}]),
    ], CleanConfig(transfer_reformulations=False))
    assert {(p["query"], p["doc_id"]) for p in off.pairs} == {("머니핀 정산", "gold")}


def test_skip_above_docs_become_hard_negatives():
    out = clean([
        _event("s1", "게이트원 설치",
               [{"doc_id": "gold", "rank": 3, "dwell_sec": 60}],
               results=["skip1", "skip2", "gold", "below"]),
    ])
    assert out.pairs[0]["negatives"] == ["skip1", "skip2"]
    assert out.report["hard_negatives"] == 2


def test_aggregation_counts_and_min_count_filter():
    events = [
        _event(f"s{i}", "노트리 템플릿", [{"doc_id": "gold", "rank": 1, "dwell_sec": 50}])
        for i in range(3)
    ] + [_event("s9", "노트리 템플릿 이상해", [{"doc_id": "rare", "rank": 1, "dwell_sec": 50}])]

    out = clean(events, CleanConfig(min_count=2))
    assert [(p["query"], p["count"]) for p in out.pairs] == [("노트리 템플릿", 3)]
    assert out.report["below_min_count"] == 1


def test_simulator_is_deterministic_and_covers_noise_kinds():
    events_a, truth_a = simulate(seed=3, config=SimConfig(n_sessions=80))
    events_b, truth_b = simulate(seed=3, config=SimConfig(n_sessions=80))
    assert events_a == events_b and truth_a == truth_b
    assert any(len(e["clicks"]) > 1 for e in events_a)              # 오클릭+정답 세션
    assert any(contains_pii(e["query"]) for e in events_a)          # PII 쿼리
    assert any(s == "jargon" for _, s in truth_a.values())          # 재검색 세션


def test_rehearsal_cleaned_beats_naive_and_recovers_jargon():
    """리허설의 결론 자체를 회귀 테스트로 고정: 정밀도 우위 + 은어 회수는 전이 규칙 몫."""
    events, truth = simulate(seed=20260720, config=SimConfig(n_sessions=300))

    def precision(pairs: list[dict]) -> float:
        unique = {(p["query"].lower(), p["doc_id"]) for p in pairs}
        good = sum(1 for q, d in unique if q in truth and truth[q][0] == d)
        return good / len(unique)

    def jargon_recall(pairs: list[dict]) -> float:
        jargon = {q for q, (_, s) in truth.items() if s == "jargon"}
        got = {p["query"].lower() for p in pairs
               if p["query"].lower() in truth and truth[p["query"].lower()][0] == p["doc_id"]}
        return len(got & jargon) / len(jargon)

    naive = naive_pairs(events)
    cleaned = clean(events).pairs
    assert precision(cleaned) > precision(naive)
    assert jargon_recall(naive) == 0.0          # 은어 쿼리에는 정답 클릭 자체가 없다
    assert jargon_recall(cleaned) > 0.6         # 전이 규칙이 회수
