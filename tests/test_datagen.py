"""Dummy data generation + JSONL round-trip — stdlib only, no torch/datasets."""
from rag.datagen.dummy import generate_dataset
from rag.datagen.synthetic import _dedup_pairs, _split_by_doc
from rag.datagen.topics import TOPICS
from rag.dataset import load_jsonl, write_jsonl


def test_split_is_schemad_disjoint_and_deterministic():
    train, test = generate_dataset(test_fraction=0.25, seed=13)
    assert train and test

    for record in train + test:
        assert set(record) == {"query", "positive"}
        assert set(record["positive"]) == {"title", "content"}

    # train/test share no records
    assert not [r for r in test if r in train]

    # ~25% test
    total = len(train) + len(test)
    assert abs(len(test) / total - 0.25) < 0.1

    # seeded → reproducible
    assert generate_dataset(test_fraction=0.25, seed=13) == (train, test)


def test_write_and_load_roundtrip(tmp_path):
    train, _ = generate_dataset()
    path = tmp_path / "train.jsonl"
    write_jsonl(str(path), train[:5])
    assert list(load_jsonl(str(path))) == train[:5]


def test_topics_train_and_eval_queries_are_disjoint():
    """Regression for 68386bc: the two phrasing pools must never overlap — a shared
    string would make the sample eval reward memorisation, not retrieval."""
    train_all = {q for t in TOPICS for q in t.train_queries}
    eval_all = {q for t in TOPICS for q in t.eval_queries}
    assert not train_all & eval_all


def test_synthetic_dedup_drops_normalised_duplicates():
    pairs = [
        {"query": "How does asyncio work?", "_doc": 0},
        {"query": "  how does ASYNCIO   work?", "_doc": 1},  # same once normalised
        {"query": "what is mypy", "_doc": 1},
    ]
    out = _dedup_pairs(pairs)
    assert [p["query"] for p in out] == ["How does asyncio work?", "what is mypy"]


def test_synthetic_split_holds_out_whole_docs():
    """A document's queries are near-paraphrases — none may straddle the split."""
    pairs = [{"query": f"q{d}-{i}", "_doc": d} for d in range(8) for i in range(3)]
    train, test = _split_by_doc(pairs, test_fraction=0.25, seed=13)

    train_docs = {p["_doc"] for p in train}
    test_docs = {p["_doc"] for p in test}
    assert not train_docs & test_docs
    assert len(test_docs) == 2                      # 25% of 8 docs
    assert len(train) + len(test) == len(pairs)     # nothing dropped
    assert _split_by_doc(pairs, 0.25, 13) == (train, test)  # seeded → deterministic

    # a single-doc corpus can't be split honestly — everything stays in train
    one = [{"query": f"q{i}", "_doc": 0} for i in range(4)]
    train1, test1 = _split_by_doc(one, 0.25, 13)
    assert len(train1) == 4 and test1 == []


def test_split_qrels_holds_out_queries_per_doc():
    from rag.datagen.eval_corpus import split_qrels

    qrels = [(f"q-{t}-{j}", f"gold-{t}", 1) for t in range(4) for j in range(3)]
    dev, final = split_qrels(qrels, final_fraction=0.3, seed=13)

    dev_q = {q for q, _, _ in dev}
    final_q = {q for q, _, _ in final}
    assert dev_q.isdisjoint(final_q)              # split BY QUERY — no leakage
    assert len(dev) + len(final) == len(qrels)
    for t in range(4):                            # stratified: every doc keeps dev queries
        assert any(q.startswith(f"q-{t}-") for q in dev_q)
        assert any(q.startswith(f"q-{t}-") for q in final_q)
    assert split_qrels(qrels, seed=13) == (dev, final)   # deterministic


def test_file_fingerprint_tracks_content(tmp_path):
    from rag.dataset import file_fingerprint

    path = tmp_path / "train.jsonl"
    path.write_text('{"a": 1}\n')
    fp = file_fingerprint(str(path))
    assert fp and len(fp) == 12
    path.write_text('{"a": 2}\n')
    assert file_fingerprint(str(path)) != fp      # regenerated in place → new hash
    assert file_fingerprint(str(tmp_path / "missing.jsonl")) is None
