"""Reciprocal Rank Fusion — the lab's hybrid (BM25 + dense) combiner."""
from __future__ import annotations

from rag.evaluation.hybrid import DEFAULT_K, fuse_dense_lexical, rrf_fuse


def test_rrf_rewards_agreement_across_rankers():
    # d1 is 2nd in BOTH lists; d2 is 1st in one but absent from the other.
    dense = {"q": ["d2", "d1"]}
    lexical = {"q": ["dx", "d1"]}
    # Small k makes the rank gaps dominate: d1 = 2/(k+2), d2 = 1/(k+1).
    fused = rrf_fuse([(1.0, dense), (1.0, lexical)], k=1)["q"]
    assert fused[0] == "d1"  # 2/3 > 1/2 — agreement beats a lone top hit


def test_weight_zero_drops_a_component():
    dense = {"q": ["d1", "d2"]}
    lexical = {"q": ["d9", "d8"]}
    # alpha=1 → lexical ignored entirely → pure dense order.
    assert fuse_dense_lexical(dense, lexical, alpha=1.0)["q"] == ["d1", "d2"]
    # alpha=0 → dense ignored → pure lexical order.
    assert fuse_dense_lexical(dense, lexical, alpha=0.0)["q"] == ["d9", "d8"]


def test_alpha_tilts_the_fusion():
    dense = {"q": ["dA", "dB"]}
    lexical = {"q": ["dB", "dA"]}
    # Symmetric disagreement: high alpha favours dense's top (dA), low favours lexical's (dB).
    assert fuse_dense_lexical(dense, lexical, alpha=0.9)["q"][0] == "dA"
    assert fuse_dense_lexical(dense, lexical, alpha=0.1)["q"][0] == "dB"


def test_missing_doc_scores_zero_not_error():
    # A query present in one component but not the other is still fused.
    fused = rrf_fuse([(0.5, {"q": ["d1"]}), (0.5, {})], k=DEFAULT_K)
    assert fused["q"] == ["d1"]


def test_depth_truncates_output():
    dense = {"q": [f"d{i}" for i in range(100)]}
    fused = fuse_dense_lexical(dense, {"q": []}, alpha=1.0, depth=10)
    assert len(fused["q"]) == 10


def test_ties_break_on_doc_id_for_reproducibility():
    # Two docs at identical ranks → deterministic order (by id), not dict insertion.
    fused = rrf_fuse([(1.0, {"q": ["dB"]}), (1.0, {"q": ["dA"]})], k=DEFAULT_K)["q"]
    assert fused == ["dA", "dB"]
