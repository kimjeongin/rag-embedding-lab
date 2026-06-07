"""Dummy data generation + JSONL round-trip — stdlib only, no torch/datasets."""
from rag.datagen.dummy import generate_dataset
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
