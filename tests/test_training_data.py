"""training.data pure helpers — no torch/datasets needed.

The behavior that must hold: ALL mined hard negatives become dataset columns for the
MNRL/GIST family (they treat every extra column as a negative), while TripletLoss
gets exactly one — and a single record without negatives drops the columns entirely
(columnar datasets can't be ragged).
"""
from rag.training.data import negative_count
from rag.training.train import resolve_matryoshka_dims


def _rows(*counts: int) -> list[dict]:
    return [{"negatives": [{"content": f"n{i}"} for i in range(c)]} for c in counts]


def test_negative_count_is_the_common_minimum():
    assert negative_count(_rows(4, 4, 4), cap=None) == 4
    assert negative_count(_rows(4, 2, 4), cap=None) == 2   # ragged → everyone drops to 2


def test_negative_count_caps_for_fixed_arity_losses():
    assert negative_count(_rows(4, 4), cap=1) == 1          # TripletLoss appetite


def test_negative_count_zero_when_any_record_lacks_negatives():
    assert negative_count(_rows(4, 0, 4), cap=None) == 0
    assert negative_count([{"query": "q"}], cap=None) == 0  # key absent entirely
    assert negative_count([], cap=None) == 0


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
