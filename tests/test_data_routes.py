"""/api/data routes added for the crawled-corpus PoC flow — crawl SSE + corpus-mode
eval set. The crawler itself is faked (its pure functions are tested in test_crawl);
these tests pin the route contracts: events out, files written, errors as 400s."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag.dataset import write_jsonl  # noqa: E402


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((event, json.loads(data)))
    return events


def test_crawl_stream_route_writes_corpus_and_streams_events(tmp_path, monkeypatch):
    import rag.datagen.crawl as crawl_mod

    async def fake_stream(url, max_pages=300, delay=0.4, min_chars=200, max_chars=2000):
        yield {"event": "start", "mode": "sitemap", "discovered": 2, "max_pages": max_pages}
        yield {"event": "page", "done": 1, "total": max_pages, "url": url, "title": "공지", "chars": 500}
        yield {"event": "done", "fetched": 2, "skipped": 1,
               "pages": [{"url": url, "title": "공지", "description": None, "content": "본문"}]}

    monkeypatch.setattr(crawl_mod, "crawl_stream", fake_stream)
    corpus_file = tmp_path / "corpus.jsonl"

    from rag.api.app import create_app

    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/data/crawl/stream",
            json={"url": "https://example.com/sitemap.xml", "max_pages": 5, "corpus_file": str(corpus_file)},
        )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert [e for e, _ in events] == ["start", "page", "done"]
    done = events[-1][1]
    assert done["count"] == 1 and done["fetched"] == 2
    assert json.loads(corpus_file.read_text())["title"] == "공지"  # written BEFORE done fired


def test_gen_eval_corpus_mode_uses_whole_corpus_as_haystack(tmp_path, monkeypatch):
    corpus_file = tmp_path / "corpus.jsonl"
    write_jsonl(str(corpus_file), [
        {"url": "u0", "title": "t0", "content": "c0"},
        {"url": "u1", "title": "t1", "content": "c1"},
    ])
    test_file = tmp_path / "test.jsonl"
    write_jsonl(str(test_file), [
        {"query": "q-a", "positive": {"title": "t1", "content": "c1"}},
        {"query": "q-b", "positive": {"title": "t0", "content": "c0"}},
    ])
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("TRAIN_EVAL_FILE", str(test_file))

    from rag.api.app import create_app

    with TestClient(create_app()) as client:
        resp = client.post("/api/data/eval", json={"source": "corpus", "corpus_file": str(corpus_file)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["corpus"] == 2 and body["queries"] == 2  # whole site = haystack

        # corpus missing → a 400 with guidance, not a 500
        missing = client.post("/api/data/eval", json={"source": "corpus", "corpus_file": str(tmp_path / "nope.jsonl")})
        assert missing.status_code == 400
        assert "없습니다" in missing.json()["detail"]


def test_label_search_reuses_the_serving_index_when_stacks_match(tmp_path, monkeypatch):
    """판정 모델 == 서빙 임베더 + 라이브 인덱스 일치 → corpus 재임베딩 없이(분 단위 절약)
    인덱스 벡터를 재사용하고, payload 내용 매칭으로 평가셋 doc id를 복원한다."""
    monkeypatch.chdir(tmp_path)
    # a saved ST model dir so infer_dim can read the pooling config (no torch)
    model_dir = tmp_path / "outputs" / "ft"
    (model_dir / "1_Pooling").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "1_Pooling" / "config.json").write_text('{"embedding_dimension": 4}')

    from rag.evaluation.beir import write_beir_dataset

    eval_dir = tmp_path / "eval"
    write_beir_dataset(
        str(eval_dir),
        corpus=[
            {"_id": "page-0", "title": "여권 재발급", "text": "온라인 신청 안내"},
            {"_id": "page-1", "title": "에너지캐시백", "text": "하반기 확대 시행"},
        ],
        queries=[{"_id": "q1", "text": "여권"}],
        qrels_rows=[("q1", "page-0", 1)],
    )
    monkeypatch.setenv("EVAL_DIR", str(eval_dir))

    from rag.api.app import create_app
    from rag.api.deps import get_store
    from rag.config import Settings
    from test_serving import FakeEmbedder, FakeStore

    store = FakeStore()
    store.create_collection("docs__outputs-ft__4d__f", dim=4)
    store.swap_alias("docs-live", "docs__outputs-ft__4d__f")
    store.query_results = [
        {"id": "1", "score": 0.9, "payload": {"url": "u1", "title": "에너지캐시백", "content": "하반기 확대 시행"}},
        {"id": "2", "score": 0.8, "payload": {"url": "u2", "title": "서빙에만 있는 문서", "content": "평가셋 밖"}},
        {"id": "3", "score": 0.7, "payload": {"url": "u0", "title": "여권 재발급", "content": "온라인 신청 안내"}},
    ]
    embedder = FakeEmbedder()

    app = create_app(Settings(
        embedder="sentence-transformers", st_model="outputs/ft", embed_dim=4,
        qdrant_collection="docs",
    ))
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as client:
        client.app.state.embedder = embedder  # process embedder singleton (fake)
        resp = client.post("/api/data/label/search", json={
            "query": "여권", "embedder": "sentence-transformers", "model": "outputs/ft",
        })

    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["results"]]
    assert ids == ["page-1", "page-0"]      # index order kept, 평가셋 밖 문서는 제외
    assert embedder.documents == []          # the corpus was NOT re-embedded
    assert embedder.queries == ["여권"]      # only the one judging query
