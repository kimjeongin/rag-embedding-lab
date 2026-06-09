"""BEIR-format IO round-trip + the in-memory ranking core (fake embedder, no network)."""
from rag.evaluation.beir import (
    load_corpus,
    load_qrels,
    load_queries,
    write_beir_dataset,
)
from rag.evaluation.retrieval import rank_corpus


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
