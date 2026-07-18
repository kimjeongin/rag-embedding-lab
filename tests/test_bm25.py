"""From-scratch BM25 — 어휘 베이스라인과 후보군 상보성(candidate_union)."""
from __future__ import annotations

import pytest

from rag.diff import candidate_union
from rag.evaluation.bm25 import BM25, rank_eval_corpus, tokenize


def test_tokenize_char_bigrams_survive_korean_josa():
    # "머니핀에서"와 "머니핀"이 bigram 수준에서 겹쳐야 조사 붙은 어절을 이긴다
    assert set(tokenize("머니핀")) <= set(tokenize("머니핀에서 정산"))
    assert tokenize("a") == ["a"]                      # 한 글자 토큰은 그대로
    assert tokenize("VPN 설치") == ["vp", "pn", "설치"]


def test_bm25_ranks_lexical_match_first_and_skips_zero_scores():
    index = BM25({
        "d1": "머니핀 법인카드 경비 정산 안내",
        "d2": "룸북 회의실 예약 안내",
        "d3": "게이트원 VPN 원격접속",
    })
    assert index.search("법인카드 정산")[0] == "d1"
    assert index.search("회의실 잡기")[0] == "d2"
    assert index.search("존재하지않는어휘조합xyz") == []   # 0점 문서는 후보가 아님


def test_rank_eval_corpus_uses_title_and_text():
    corpus = {
        "p1": {"title": "룸북 사용 가이드", "text": "회의실 예약 절차"},
        "p2": {"title": "메일가드", "text": "스팸 차단"},
    }
    rankings = rank_eval_corpus(corpus, {"q1": "룸북 예약"}, top_k=2)
    assert rankings["q1"][0] == "p1"


def _run(run_id: str, rankings: dict[str, list[str]], fingerprint: str = "fp1") -> dict:
    return {
        "id": run_id,
        "eval_fingerprint": fingerprint,
        "rankings": rankings,
        "per_query": {q: {"ndcg@10": 0.0} for q in rankings},
    }


def test_candidate_union_measures_marginal_contribution():
    qrels = {"q1": {"g1": 1.0}, "q2": {"g2": 1.0}, "q3": {"g3": 1.0}}
    bm25 = _run("a", {"q1": ["g1"], "q2": ["x"], "q3": ["y"]})       # q1만 잡음
    dense = _run("b", {"q1": ["g1"], "q2": ["g2"], "q3": ["z"]})     # q1(중복) + q2(신규)

    out = candidate_union(bm25, dense, qrels, k=10)
    assert out["n"] == 3
    assert out["recall_a"] == pytest.approx(1 / 3)
    assert out["recall_b"] == pytest.approx(2 / 3)
    assert out["recall_union"] == pytest.approx(2 / 3)
    assert out["marginal_b"] == pytest.approx(1 / 3)   # dense가 실제로 보탠 건 q2 하나
    assert out["marginal_a"] == pytest.approx(0.0)     # BM25는 union에 새로 보탠 게 없음


def test_candidate_union_slices_and_guards():
    qrels = {"q1": {"g1": 1.0}, "q2": {"g2": 1.0}}
    a = _run("a", {"q1": ["g1"], "q2": ["x"]})
    b = _run("b", {"q1": ["x"], "q2": ["g2"]})

    out = candidate_union(a, b, qrels, slice_map={"q1": "standard", "q2": "jargon"})
    by_name = {s["topic"]: s for s in out["slices"]}
    assert by_name["jargon"]["marginal_b"] == pytest.approx(1.0)
    assert by_name["standard"]["marginal_b"] == pytest.approx(0.0)

    with pytest.raises(ValueError):
        candidate_union(a, _run("b", {}, fingerprint="other"), qrels)
    assert candidate_union(a, _run("b", {}), qrels) is None          # 랭킹 없는 레거시 런
