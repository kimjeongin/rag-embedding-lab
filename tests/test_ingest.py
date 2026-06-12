"""Real-data ingestion — log parsing, pair/qrels transforms, TREC runs, the routes."""
from __future__ import annotations

import pytest

from rag.datagen.ingest import dedupe_pairs, parse_records, to_qrels, to_train_pairs
from rag.evaluation.trec import parse_trec_run

_CORPUS = {
    "site-1": {"title": "VPN 가이드", "text": "사내 VPN 설정"},
    "site-2": {"title": "휴가 포털", "text": "연차 신청"},
}


def test_parse_records_auto_detects_jsonl_and_csv():
    jsonl, errors = parse_records('{"query": "vpn 안됨", "doc_id": "site-1"}\nnot json')
    assert jsonl == [{"query": "vpn 안됨", "doc_id": "site-1"}]
    assert errors and "2행" in errors[0]

    with_header, _ = parse_records("query,doc_id\n연차,site-2\n")
    assert with_header == [{"query": "연차", "doc_id": "site-2"}]

    headerless, _ = parse_records("vpn 안됨,site-1")            # click-log export shape
    assert headerless == [{"query": "vpn 안됨", "doc_id": "site-1"}]


def test_to_train_pairs_resolves_doc_ids_and_keeps_inline_content():
    records = [
        {"query": "vpn 안됨", "doc_id": "site-1"},
        {"query": "보안 교육", "title": "교육 안내", "content": "연 1회 필수"},
        {"query": "유령", "doc_id": "missing"},
        {"query": ""},
    ]
    pairs, skipped = to_train_pairs(records, _CORPUS)
    assert pairs[0]["positive"]["title"] == "VPN 가이드"        # corpus join
    assert pairs[1]["positive"]["content"] == "연 1회 필수"     # inline content
    assert len(skipped) == 2

    fresh = dedupe_pairs(pairs, [{"query": "VPN 안됨", "positive": {"title": None, "content": "사내 VPN 설정"}}])
    assert fresh == []                                          # case-insensitive query dedupe


def test_to_qrels_merges_same_query_text_and_skips_unknown_docs():
    records = [
        {"query": "vpn 안됨", "doc_id": "site-1"},
        {"query": "VPN 안됨", "doc_id": "site-2"},              # same query, second relevant doc
        {"query": "외부", "doc_id": "nope"},
    ]
    new_queries, rows, skipped = to_qrels(records, _CORPUS, taken_query_ids={"q-user-1"})
    assert len(new_queries) == 1                                # one query record for both rows
    assert new_queries[0]["_id"] == "q-user-2"                  # taken id skipped
    assert {(r[0], r[1]) for r in rows} == {("q-user-2", "site-1"), ("q-user-2", "site-2")}
    assert len(skipped) == 1


def test_parse_trec_run_orders_by_rank_and_dedupes():
    text = "\n".join(
        [
            "q1 Q0 d2 2 8.1 bm25",
            "q1 Q0 d1 1 9.9 bm25",
            "q1 Q0 d1 3 7.0 bm25",                              # duplicate doc → kept once
            "broken line",
        ]
    )
    rankings, errors = parse_trec_run(text)
    assert rankings == {"q1": ["d1", "d2"]}
    assert len(errors) == 1


def test_import_routes_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag.dataset import load_jsonl
    from rag.evaluation.beir import load_qrels, write_beir_dataset

    eval_dir = tmp_path / "eval"
    write_beir_dataset(
        str(eval_dir),
        [{"_id": "site-1", "title": "VPN 가이드", "text": "vpn"}, {"_id": "site-2", "title": "휴가", "text": "pto"}],
        [{"_id": "q-0-0", "text": "기존 쿼리"}],
        [("q-0-0", "site-1", 1)],
    )
    monkeypatch.setenv("EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("TRAIN_FILE", str(tmp_path / "train.jsonl"))
    monkeypatch.setenv("TRAIN_EVAL_FILE", str(tmp_path / "test.jsonl"))
    monkeypatch.setenv("RUNS_FILE", str(tmp_path / "evals.jsonl"))

    from rag.api.app import create_app

    with TestClient(create_app()) as client:
        # click-log import → both train pairs and qrels
        resp = client.post(
            "/api/data/import",
            json={"content": "query,doc_id\nvpn 안됨,site-1\n연차 어디서,site-2\n", "target": "both"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["added_train"] == 2 and body["added_qrels"] == 2
        assert body["fingerprint_changed"] is True

        pairs = list(load_jsonl(str(tmp_path / "train.jsonl")))
        assert pairs[0]["positive"]["title"] == "VPN 가이드"
        qrels = load_qrels(str(eval_dir), "test")               # legacy single-split layout
        assert any(q.startswith("q-user-") for q in qrels)

        # judging loop: commit clicked docs for a query
        commit = client.post(
            "/api/data/label/commit",
            json={"query": "wifi 비번", "doc_ids": ["site-1"], "also_train": True},
        )
        assert commit.status_code == 200
        assert commit.json()["added_qrels"] == 1

        # external BM25 ranking → ordinary registry run
        trec = client.post(
            "/api/runs/import-trec",
            json={"label": "BM25 (production)", "content": "q-0-0 Q0 site-1 1 9.9 bm25"},
        )
        assert trec.status_code == 200
        run = trec.json()["run"]
        assert run["embedder"] == "external"
        assert trec.json()["metrics"]["recall@1"] == 1.0
