"""BEIR-format IO round-trip + the in-memory ranking core (fake embedder, no network)."""
import contextlib

from rag.evaluation.beir import (
    eval_set_fingerprint,
    load_corpus,
    load_qrels,
    load_queries,
    write_beir_dataset,
)
from rag.evaluation.retrieval import evaluate, rank_corpus


def _write_sample(eval_dir) -> None:
    corpus = [
        {"_id": "d1", "title": "Cats", "text": "cats are small domestic felines"},
        {"_id": "d2", "title": "Dogs", "text": "dogs are loyal domestic canines"},
        {"_id": "d3", "title": "Cars", "text": "cars are road vehicles with engines"},
    ]
    queries = [{"_id": "q1", "text": "felines"}, {"_id": "q2", "text": "engines"}]
    qrels = [("q1", "d1", 1), ("q2", "d3", 1)]
    write_beir_dataset(str(eval_dir), corpus, queries, qrels)


def test_beir_io_roundtrip(tmp_path):
    _write_sample(tmp_path)
    corpus = load_corpus(str(tmp_path))
    queries = load_queries(str(tmp_path))
    qrels = load_qrels(str(tmp_path))

    assert corpus["d1"]["title"] == "Cats"
    assert corpus["d1"]["text"].startswith("cats")
    assert queries == {"q1": "felines", "q2": "engines"}
    assert qrels == {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}   # header skipped, score>0 kept


class _KeywordEmbedder:
    """Bag-of-words vectors over a fixed vocab — deterministic, no model or network."""

    _VOCAB = ("felines", "canines", "engines", "domestic", "vehicles", "cats", "dogs", "cars")

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(word in lowered) for word in self._VOCAB]

    async def embed_documents(self, documents):
        return [self._vec(f"{d.title or ''} {d.content}") for d in documents]

    async def embed_queries(self, queries):
        return [self._vec(q) for q in queries]


async def test_rank_corpus_orders_by_cosine(tmp_path):
    _write_sample(tmp_path)
    corpus = load_corpus(str(tmp_path))
    queries = load_queries(str(tmp_path))

    rankings = await rank_corpus(_KeywordEmbedder(), corpus, queries)

    assert rankings["q1"][0] == "d1"   # "felines" → the Cats doc
    assert rankings["q2"][0] == "d3"   # "engines" → the Cars doc


def test_eval_set_fingerprint_tracks_content_not_path(tmp_path):
    _write_sample(tmp_path)
    fp = eval_set_fingerprint(str(tmp_path))
    assert fp and len(fp) == 12
    assert eval_set_fingerprint(str(tmp_path)) == fp   # same contents → same hash

    # regenerate IN PLACE with different contents — same path, different fingerprint
    write_beir_dataset(
        str(tmp_path),
        [{"_id": "d1", "title": "Cats", "text": "rewritten corpus"}],
        [{"_id": "q1", "text": "felines"}],
        [("q1", "d1", 1)],
    )
    assert eval_set_fingerprint(str(tmp_path)) != fp

    # incomplete set → None (nothing to identify)
    assert eval_set_fingerprint(str(tmp_path / "missing")) is None


async def test_evaluate_end_to_end_reports_per_query_and_ci(tmp_path, monkeypatch):
    """evaluate() glues loading, ranking and scoring — run it whole with a fake embedder."""
    import rag.embeddings as embeddings
    from rag.config import Settings

    @contextlib.asynccontextmanager
    async def fake_build(_settings):
        yield _KeywordEmbedder()

    monkeypatch.setattr(embeddings, "build_embedder", fake_build)
    _write_sample(tmp_path)

    report = await evaluate(Settings.from_env(), str(tmp_path))

    assert report.metrics["recall@1"] == 1.0           # both queries hit at rank 1
    assert set(report.per_query) == {"q1", "q2"}
    lo, hi = report.ci95["ndcg@10"]
    assert lo <= report.metrics["ndcg@10"] <= hi
