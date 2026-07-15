"""Judging-loop corpus cache — embed the corpus once, then one query per judgment."""
from __future__ import annotations

import pytest

from rag.api import labelrank
from rag.config import Settings


class FakeEmbedder:
    """Two-dim vectors: d1→x축, d2→y축 — a query of [0,1] must rank d2 first."""

    def __init__(self) -> None:
        self.doc_calls = 0
        self.query_calls = 0

    async def embed_documents(self, documents):
        self.doc_calls += 1
        axes = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        return axes[: len(documents)]

    async def embed_queries(self, queries):
        self.query_calls += 1
        return [[0.0, 1.0] for _ in queries]


CORPUS = {
    "d1": {"title": "엑스", "text": "x"},
    "d2": {"title": "와이", "text": "y"},
    "d3": {"title": "사선", "text": "xy"},
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    labelrank._entry = None
    yield
    labelrank._entry = None


async def test_rank_orders_by_cosine_and_caches_the_doc_matrix():
    fake = FakeEmbedder()
    settings = Settings(embed_dim=2)

    first = await labelrank.rank(settings, CORPUS, "fp-a", "쿼리", shared=fake)
    assert first[0] == "d2"                      # [0,1] query → y-axis doc wins
    assert first == ["d2", "d3", "d1"]

    await labelrank.rank(settings, CORPUS, "fp-a", "다른 쿼리", shared=fake)
    assert fake.doc_calls == 1                   # corpus embedded ONCE
    assert fake.query_calls == 2                 # one per judgment


async def test_corpus_change_invalidates_the_cache():
    fake = FakeEmbedder()
    settings = Settings(embed_dim=2)

    await labelrank.rank(settings, CORPUS, "fp-a", "쿼리", shared=fake)
    await labelrank.rank(settings, CORPUS, "fp-b", "쿼리", shared=fake)  # regenerated corpus
    assert fake.doc_calls == 2


async def test_model_change_evicts_and_rebuilds():
    fake_a, fake_b = FakeEmbedder(), FakeEmbedder()
    await labelrank.rank(Settings(embed_dim=2), CORPUS, "fp-a", "쿼리", shared=fake_a)
    await labelrank.rank(
        Settings(st_model="outputs/other", embed_dim=2), CORPUS, "fp-a", "쿼리", shared=fake_b
    )
    assert fake_a.doc_calls == 1
    assert fake_b.doc_calls == 1                 # new entry, not reused


async def test_top_n_bounds_the_result():
    ranked = await labelrank.rank(
        Settings(embed_dim=2), CORPUS, "fp-a", "쿼리", top_n=2, shared=FakeEmbedder()
    )
    assert len(ranked) == 2


def test_map_hits_recovers_doc_ids_by_content():
    hits = [
        {"score": 0.9, "title": "와이", "content": "y", "url": "https://x/y"},
        {"score": 0.8, "title": "밖의 문서", "content": "eval corpus에 없음", "url": None},
        {"score": 0.7, "title": "엑스", "content": "x", "url": None},
        {"score": 0.6, "title": "와이", "content": "y", "url": None},  # duplicate content
    ]
    assert labelrank.map_hits_to_corpus_ids(hits, CORPUS) == ["d2", "d1"]


def test_map_hits_empty_when_contents_drifted():
    hits = [{"score": 0.9, "title": "새 크롤", "content": "재크롤로 내용이 바뀜"}]
    assert labelrank.map_hits_to_corpus_ids(hits, CORPUS) == []


def test_map_hits_respects_top_n():
    hits = [
        {"title": "엑스", "content": "x"},
        {"title": "와이", "content": "y"},
        {"title": "사선", "content": "xy"},
    ]
    assert labelrank.map_hits_to_corpus_ids(hits, CORPUS, top_n=2) == ["d1", "d2"]
