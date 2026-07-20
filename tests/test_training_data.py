"""training.data pure helpers — no torch/datasets needed.

The behavior that must hold: ALL mined hard negatives become dataset columns for the
MNRL/GIST family (they treat every extra column as a negative), while TripletLoss
gets exactly one — and records with fewer negatives than the target are PADDED from
other records' docs (columnar datasets can't be ragged, and the old take-the-minimum
rule let one zero-negative record silently discard every mined negative).
"""
from rag.training.data import negative_count, pad_negatives, to_ir_eval
from rag.training.train import resolve_matryoshka_dims


def _rows(*counts: int) -> list[dict]:
    return [
        {
            "positive": {"title": f"p{j}", "content": f"pc{j}"},
            "negatives": [{"title": f"n{j}-{i}", "content": f"nc{j}-{i}"} for i in range(c)],
        }
        for j, c in enumerate(counts)
    ]


def test_negative_count_is_the_largest_supply():
    assert negative_count(_rows(4, 4, 4), cap=None) == 4
    assert negative_count(_rows(4, 2, 4), cap=None) == 4   # short rows get padded, not dropped
    assert negative_count(_rows(4, 0, 4), cap=None) == 4


def test_negative_count_caps_for_fixed_arity_losses():
    assert negative_count(_rows(4, 4), cap=1) == 1          # TripletLoss appetite


def test_negative_count_zero_when_no_record_has_negatives():
    assert negative_count(_rows(0, 0), cap=None) == 0
    assert negative_count([{"query": "q"}], cap=None) == 0  # key absent entirely
    assert negative_count([], cap=None) == 0


def test_pad_negatives_fills_short_rows_from_other_records():
    rows = _rows(3, 0, 1)
    padded = pad_negatives(rows, target=3, seed=13)
    assert padded == 2                                       # rows 1 and 2 were short
    for row in rows:
        assert len(row["negatives"]) == 3
        # never pad a record with its own positive — that would teach the model
        # to push the right answer away
        own = (row["positive"]["title"], row["positive"]["content"])
        assert own not in {(n["title"], n["content"]) for n in row["negatives"]}
    # deterministic given the seed
    rows2 = _rows(3, 0, 1)
    pad_negatives(rows2, target=3, seed=13)
    assert rows2 == rows


def test_pad_negatives_leaves_full_rows_untouched():
    rows = _rows(2, 2)
    before = [list(r["negatives"]) for r in rows]
    assert pad_negatives(rows, target=2) == 0
    assert [r["negatives"] for r in rows] == before


def test_to_ir_eval_adds_distractor_docs_to_the_corpus(tmp_path):
    import json

    def _write(path, records):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    eval_file = tmp_path / "test.jsonl"
    train_file = tmp_path / "train.jsonl"
    _write(eval_file, [
        {"query": "q-a", "positive": {"title": "t0", "content": "c0"}},
        {"query": "q-b", "positive": {"title": "t0", "content": "c0"}},  # same doc, deduped
    ])
    _write(train_file, [
        {"query": "q-c", "positive": {"title": "t1", "content": "c1"}},
        {"query": "q-d", "positive": {"title": "t0", "content": "c0"}},  # already in corpus
    ])

    queries, corpus, relevant = to_ir_eval(str(eval_file), "task", str(train_file))
    assert len(queries) == 2
    assert len(corpus) == 2                                  # t0 + the t1 distractor, deduped
    gold = {doc_id for ids in relevant.values() for doc_id in ids}
    assert len(gold) == 1                                    # distractors are nobody's answer

    # without distractors the corpus is just the eval docs
    _, corpus_plain, _ = to_ir_eval(str(eval_file), "task")
    assert len(corpus_plain) == 1


def test_resolve_matryoshka_dims_auto_halves_from_model_dim():
    # no request → full dim, then halve down to ≥64
    assert resolve_matryoshka_dims((), 1024) == [1024, 512, 256, 128, 64]
    assert resolve_matryoshka_dims((), 768) == [768, 384, 192, 96]


def test_resolve_matryoshka_dims_honours_a_request_but_clamps_to_model_dim():
    assert resolve_matryoshka_dims((256, 128, 64), 1024) == [256, 128, 64]
    assert resolve_matryoshka_dims((9999, 256), 1024) == [256]       # > model dim is dropped
    assert resolve_matryoshka_dims((128, 256, 128), 1024) == [256, 128]  # sorted desc, deduped


def test_resolve_matryoshka_dims_falls_back_to_auto_when_request_unusable():
    # every requested dim exceeds the model → ignore the request, derive from the dim
    assert resolve_matryoshka_dims((9999,), 768) == [768, 384, 192, 96]


def test_negative_count_cap_zero_drops_all_columns():
    # max_negatives=0: mined negatives가 있어도 컬럼을 만들지 않는다 (in-batch만 —
    # negatives가 배치 메모리를 배로 늘려 OOM이 날 때 쓰는 knob).
    assert negative_count(_rows(4, 2), cap=0) == 0
