"""Pure data helpers behind the web UI.

These need pandas (the `ui` dependency group); the test is skipped when it isn't
installed, so the core test suite still runs with minimal deps.
"""
import pytest

pytest.importorskip("pandas")

from rag import runs  # noqa: E402 — after importorskip by design
from rag.webui import actions  # noqa: E402


def test_default_model_prefers_embedding_for_ollama():
    assert actions.default_model("ollama", ["qwen3:4b", "qwen3-embedding:0.6b"]) == "qwen3-embedding:0.6b"
    assert actions.default_model("ollama", ["a", "b"]) == "a"  # no embedding → first
    assert actions.default_model("sentence-transformers", ["outputs/ft", "x"]) == "outputs/ft"
    assert actions.default_model("ollama", []) == ""


def test_compare_all_data_is_long_form_and_zoomed(tmp_path, monkeypatch):
    path = str(tmp_path / "runs.jsonl")
    monkeypatch.setenv("RUNS_FILE", path)
    runs.append_run("base", "ollama", "m", "data/eval", {"recall@1": 0.9, "ndcg@10": 0.95}, path=path)

    df, y_lim = actions.compare_all_data()
    assert list(df.columns) == ["metric", "run", "value"]
    assert len(df) == 2  # one row per present (metric, run) pair
    assert y_lim[1] == 1.0 and 0.0 <= y_lim[0] < 1.0  # y-axis zoomed to the data, capped at 1
