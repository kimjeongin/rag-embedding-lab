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


def test_round_trip_filter_keeps_only_self_retrieving_pairs():
    """Promptagator consistency filter: a pair survives iff its query ranks its own
    source doc in the top-k — generated queries that retrieve some OTHER page are
    noise, not supervision."""
    import numpy as np

    from rag.datagen.synthetic import _round_trip_keep

    sims = np.array([
        [0.9, 0.2],   # pair0 → doc0: top-1 is doc0 ✓
        [0.3, 0.8],   # pair1 → doc0: top-1 is doc1 ✗
        [0.7, 0.6],   # pair2 → doc1: top-1 is doc0 ✗
    ])
    assert _round_trip_keep(sims, [0, 0, 1], k=1) == [0]
    assert _round_trip_keep(sims, [0, 0, 1], k=2) == [0, 1, 2]  # k≥n_docs keeps all


def test_attach_negatives_skips_probable_false_negatives():
    """TopK-PercPos guard: a candidate scoring within the margin of the positive is
    presumed to be a true answer wearing the wrong label — skipped, not trained on."""
    import numpy as np

    from rag.datagen.synthetic import _attach_negatives

    docs = [{"title": f"d{i}", "content": f"c{i}"} for i in range(4)]
    sims = np.array([[0.8, 0.79, 0.5, 0.3]])  # doc1 ≈ the positive → probable false neg
    pairs = [{"query": "q", "_doc": 0}]
    _attach_negatives(pairs, sims, docs, n_negatives=2, margin=0.05)
    assert [n["title"] for n in pairs[0]["negatives"]] == ["d2", "d3"]

    pairs = [{"query": "q", "_doc": 0}]
    _attach_negatives(pairs, sims, docs, n_negatives=2, margin=0.0)  # guard off
    assert [n["title"] for n in pairs[0]["negatives"]] == ["d1", "d2"]


def test_eval_from_corpus_builds_beir_set_and_skips_stale_pairs():
    from rag.datagen.eval_from_corpus import build

    docs = [
        {"url": "u0", "title": "t0", "content": "c0"},
        {"url": "u1", "title": "t1", "content": "c1"},
    ]
    pairs = [
        {"query": "q-a", "positive": {"title": "t1", "content": "c1"}},
        {"query": "q-b", "positive": {"title": "tX", "content": "cX"}},  # stale → skipped
    ]
    corpus, queries, qrels, skipped = build(docs, pairs)
    assert [d["_id"] for d in corpus] == ["page-0", "page-1"]   # whole site = haystack
    assert corpus[1] == {"_id": "page-1", "title": "t1", "text": "c1"}
    assert queries == [{"_id": "q-test-0", "text": "q-a"}]
    assert qrels == [("q-test-0", "page-1", 1)]
    assert skipped == 1


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
