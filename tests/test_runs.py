"""Eval-run registry — append/load/delete round-trip (stdlib only)."""
from rag.runs import (
    METRIC_KEYS,
    append_run,
    best_per_metric,
    delete_run,
    load_runs,
)


def test_append_and_load_newest_first(tmp_path):
    path = str(tmp_path / "evals.jsonl")
    base_metrics = {k: 0.5 for k in METRIC_KEYS}
    a = append_run("base", "ollama", "qwen3", "data/eval", base_metrics, path=path)
    b = append_run(
        "ft", "sentence-transformers", "outputs/ft", "data/eval",
        {**base_metrics, "recall@1": 0.9}, path=path,
    )

    runs = load_runs(path)
    assert [r["id"] for r in runs] == [b["id"], a["id"]]   # newest first
    assert runs[0]["label"] == "ft"
    assert runs[0]["metrics"]["recall@1"] == 0.9
    assert runs[1]["embedder"] == "ollama"


def test_blank_label_falls_back_to_model(tmp_path):
    path = str(tmp_path / "e.jsonl")
    record = append_run("  ", "ollama", "qwen3-embedding:0.6b", "data/eval", {"recall@1": 1.0}, path=path)
    assert record["label"] == "qwen3-embedding:0.6b"


def test_load_missing_file_returns_empty(tmp_path):
    assert load_runs(str(tmp_path / "nope.jsonl")) == []


def test_delete_run_removes_only_that_id(tmp_path):
    path = str(tmp_path / "e.jsonl")
    a = append_run("a", "ollama", "m1", "data/eval", {"recall@1": 0.4}, path=path)
    b = append_run("b", "ollama", "m2", "data/eval", {"recall@1": 0.6}, path=path)
    remaining = delete_run(a["id"], path=path)
    assert remaining == 1
    assert [r["id"] for r in load_runs(path)] == [b["id"]]


def test_best_per_metric_takes_the_max(tmp_path):
    path = str(tmp_path / "e.jsonl")
    append_run("a", "ollama", "m1", "data/eval", {"recall@1": 0.4, "ndcg@10": 0.9}, path=path)
    append_run("b", "ollama", "m2", "data/eval", {"recall@1": 0.7, "ndcg@10": 0.8}, path=path)
    best = best_per_metric(path)
    assert best["recall@1"] == 0.7
    assert best["ndcg@10"] == 0.9


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    """An append torn by a crash must not take down the whole registry."""
    path = tmp_path / "e.jsonl"
    a = append_run("a", "ollama", "m1", "data/eval", {"recall@1": 0.4}, path=str(path))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "torn", "metr\n')  # partial write, e.g. power loss mid-append
    b = append_run("b", "ollama", "m2", "data/eval", {"recall@1": 0.6}, path=str(path))

    runs = load_runs(str(path))
    assert [r["id"] for r in runs] == [b["id"], a["id"]]  # both intact rows survive

    # delete rewrites the file — the corrupt line is dropped, the survivor kept
    assert delete_run(a["id"], path=str(path)) == 1
    assert [r["id"] for r in load_runs(str(path))] == [b["id"]]


def test_append_run_records_eval_set_identity(tmp_path):
    path = str(tmp_path / "e.jsonl")
    append_run(
        "a", "ollama", "m", "data/eval", {"ndcg@10": 0.9}, path=path,
        eval_fingerprint="abc123def456", n_queries=48,
        ci95={"ndcg@10": (0.85, 0.95)}, per_query={"q1": {"ndcg@10": 0.9}},
    )
    loaded = load_runs(path)[0]
    assert loaded["eval_fingerprint"] == "abc123def456"
    assert loaded["n_queries"] == 48
    assert loaded["ci95"]["ndcg@10"] == [0.85, 0.95]
    assert loaded["per_query"]["q1"]["ndcg@10"] == 0.9


def test_best_per_metric_scopes_by_fingerprint(tmp_path):
    """Scores from different eval-set contents must not mix into one 'best'."""
    path = str(tmp_path / "e.jsonl")
    append_run("old", "ollama", "m1", "data/eval", {"ndcg@10": 0.99}, path=path,
               eval_fingerprint="old-set")
    append_run("new", "ollama", "m2", "data/eval", {"ndcg@10": 0.80}, path=path,
               eval_fingerprint="new-set")
    append_run("legacy", "ollama", "m3", "data/eval", {"ndcg@10": 0.95}, path=path)  # pre-fingerprint

    assert best_per_metric(path, fingerprint="new-set") == {"ndcg@10": 0.80}
    assert best_per_metric(path)["ndcg@10"] == 0.99  # unscoped: max over everything
