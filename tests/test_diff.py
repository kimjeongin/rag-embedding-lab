"""Paired run-vs-run comparison — permutation test + win/loss + the HTTP joins."""
from __future__ import annotations

import pytest

from rag.diff import compare_runs, paired_permutation_test


def _run(run_id: str, scores: dict[str, float], fingerprint: str = "fp1", **extra) -> dict:
    return {
        "id": run_id,
        "created_at": "2026-06-11T00:00:00",
        "label": run_id,
        "embedder": "sentence-transformers",
        "model": f"outputs/{run_id}",
        "eval_dir": "data/eval",
        "metrics": {"ndcg@10": sum(scores.values()) / len(scores)},
        "eval_fingerprint": fingerprint,
        "per_query": {q: {"ndcg@10": s} for q, s in scores.items()},
        **extra,
    }


def test_permutation_test_is_sane():
    assert paired_permutation_test([]) == 1.0
    assert paired_permutation_test([0.0] * 10) == 1.0           # identical runs → no evidence
    consistent = [0.1] * 20                                      # every query improved
    assert paired_permutation_test(consistent) < 0.01
    noise = [0.1, -0.1] * 10                                     # perfectly mixed → no signal
    assert paired_permutation_test(noise) > 0.5


def test_compare_runs_counts_wins_and_orders_regressions_first():
    a = _run("a", {"q-0-0": 0.5, "q-0-1": 0.5, "q-1-0": 0.9, "q-1-1": 0.2})
    b = _run("b", {"q-0-0": 0.8, "q-0-1": 0.5, "q-1-0": 0.7, "q-1-1": 0.6})

    result = compare_runs(a, b)

    assert (result["wins"], result["losses"], result["ties"]) == (2, 1, 1)
    assert result["queries"][0]["query_id"] == "q-1-0"           # worst regression first
    assert result["delta"] == pytest.approx(result["mean_b"] - result["mean_a"])
    assert 0.0 < result["p_value"] <= 1.0
    assert "ndcg@10" in result["by_metric"]
    topics = {s["topic"] for s in result["slices"]}
    assert topics == {"0", "1"}                                  # per-topic slices


def test_compare_runs_refuses_different_eval_sets():
    a = _run("a", {"q1": 0.5})
    b = _run("b", {"q1": 0.6}, fingerprint="other")
    with pytest.raises(ValueError):
        compare_runs(a, b)
    with pytest.raises(ValueError):
        compare_runs(_run("a", {"q1": 0.5}, fingerprint=None), _run("b", {"q1": 0.6}, fingerprint=None))


def test_diff_route_joins_texts_when_eval_set_matches(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag import runs as registry
    from rag.evaluation.beir import eval_set_fingerprint, write_beir_dataset

    eval_dir = tmp_path / "eval"
    write_beir_dataset(
        str(eval_dir),
        [{"_id": "d1", "title": "VPN 가이드", "text": "vpn"}, {"_id": "d2", "title": "휴가 신청", "text": "pto"}],
        [{"_id": "q-0-0", "text": "vpn 안됨"}, {"_id": "q-0-1", "text": "연차 쓰는 법"}],
        [("q-0-0", "d1", 1), ("q-0-1", "d2", 1)],
    )
    fp = eval_set_fingerprint(str(eval_dir))
    runs_file = tmp_path / "evals.jsonl"
    monkeypatch.setenv("RUNS_FILE", str(runs_file))

    base = dict(eval_fingerprint=fp, split="test")
    rec_a = registry.append_run(
        "a", "st", "outputs/a", str(eval_dir), {"ndcg@10": 0.5},
        per_query={"q-0-0": {"ndcg@10": 0.0}, "q-0-1": {"ndcg@10": 1.0}},
        rankings={"q-0-0": ["d2", "d1"], "q-0-1": ["d2"]}, **base,
    )
    rec_b = registry.append_run(
        "b", "st", "outputs/b", str(eval_dir), {"ndcg@10": 1.0},
        per_query={"q-0-0": {"ndcg@10": 1.0}, "q-0-1": {"ndcg@10": 1.0}},
        rankings={"q-0-0": ["d1", "d2"], "q-0-1": ["d2"]}, **base,
    )

    from rag.api.app import create_app

    with TestClient(create_app()) as client:
        resp = client.get("/api/runs/diff", params={"a": rec_a["id"], "b": rec_b["id"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["texts_available"] is True
        first = body["queries"][0]
        assert first["text"] in ("vpn 안됨", "연차 쓰는 법")
        assert first["retrieved_a"][0]["title"]                  # doc titles joined
        assert any(d["relevant"] for d in first["retrieved_b"])  # relevance flags present

        missing = client.get("/api/runs/diff", params={"a": rec_a["id"], "b": "nope"})
        assert missing.status_code == 404
