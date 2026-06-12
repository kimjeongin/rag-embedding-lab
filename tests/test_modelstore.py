"""Saved-model shelf — detail joins, guarded delete, handoff marker (no torch)."""
from __future__ import annotations

import json

import pytest

from rag import modelstore


def _make_model(root, name: str, meta: dict | None = None, dim: int = 1024):
    path = root / "outputs" / name
    (path / "1_Pooling").mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "1_Pooling" / "config.json").write_text(json.dumps({"embedding_dimension": dim}))
    (path / "model.bin").write_bytes(b"x" * 100)
    if meta:
        (path / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return f"outputs/{name}"


def test_model_detail_joins_meta_and_eval_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNS_FILE", str(tmp_path / "evals.jsonl"))
    from rag import runs as registry

    path = _make_model(tmp_path, "m1", meta={"loss": "mnrl", "saved_epoch": 7, "note": "기준 런"})
    registry.append_run("dev했", "st", path, "data/eval", {"ndcg@10": 0.90}, split="dev", n_queries=30)
    registry.append_run("더 좋음", "st", path, "data/eval", {"ndcg@10": 0.95}, split="dev", n_queries=30)
    registry.append_run("최종", "st", path, "data/eval", {"ndcg@10": 0.93}, split="final", n_queries=14)

    detail = modelstore.model_detail(path)

    assert detail["dim"] == 1024
    assert detail["size_bytes"] > 0
    assert detail["meta"]["loss"] == "mnrl"
    assert detail["eval_dev"]["metrics"]["ndcg@10"] == 0.95     # BEST dev run, not latest
    assert detail["eval_final"]["metrics"]["ndcg@10"] == 0.93
    assert detail["handed_off"] is False

    listing = modelstore.list_detail([path])
    assert listing["disk_total_bytes"] == detail["size_bytes"]


def test_delete_model_is_guarded_to_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _make_model(tmp_path, "m1")

    with pytest.raises(ValueError):
        modelstore.delete_model(str(tmp_path / "elsewhere"))    # outside outputs/
    with pytest.raises(ValueError):
        modelstore.delete_model("outputs/none")                 # missing

    modelstore.delete_model(path)
    assert not (tmp_path / "outputs" / "m1").exists()


def test_handed_off_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert modelstore.handed_off_model() is None
    marker = tmp_path / "runs" / "handoff.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"model": "outputs/m1", "at": "2026-06-11T12:00:00"}')
    assert modelstore.handed_off_model()["model"] == "outputs/m1"
