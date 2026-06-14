"""Cross-run median-pruning decision — the pure helpers in rag.api.jobs.

A sweep run is pruned when its best-so-far validation nDCG trails the median of the
runs that already finished, judged at the same epoch. The decision is held off until
a couple of epochs in (early curves are noisy) and until enough peers exist to form a
median. No torch / subprocess needed — these are plain functions over the epoch points
the runner already streams.
"""
from rag.api.jobs import _PRUNE_MIN_EPOCH, _PRUNE_WARMUP, _best_ndcg_at, _peer_bests_at, _should_prune


def _epochs(*ndcgs):
    """[(epoch, ndcg), …] → the run-state epoch dicts (None = metric missing)."""
    return [{"epoch": i + 1, "ndcg": v} for i, v in enumerate(ndcgs)]


def test_best_ndcg_at_is_the_running_max_through_an_epoch():
    epochs = _epochs(0.50, 0.42, 0.61, 0.58)
    assert _best_ndcg_at(epochs, 2) == 0.50      # max(0.50, 0.42)
    assert _best_ndcg_at(epochs, 4) == 0.61
    assert _best_ndcg_at(_epochs(None, None), 2) is None
    assert _best_ndcg_at([], 3) is None


def test_should_prune_only_when_trailing_the_peer_median():
    peers = [0.50, 0.55, 0.60]                   # median 0.55
    assert _should_prune(0.40, peers, epoch=3) is True
    assert _should_prune(0.58, peers, epoch=3) is False   # above median → keep


def test_should_prune_holds_off_early_and_without_enough_peers():
    peers = [0.50, 0.55, 0.60]
    assert _should_prune(0.10, peers, epoch=_PRUNE_MIN_EPOCH - 1) is False  # too early
    too_few = [0.55] * (_PRUNE_WARMUP - 1)
    assert _should_prune(0.10, too_few, epoch=5) is False                    # not enough peers
    assert _should_prune(None, peers, epoch=5) is False                      # no metric yet


def test_peer_bests_skips_self_and_unfinished_runs():
    job = {
        "runs": [
            {"idx": 0, "status": "evaluated", "epochs": _epochs(0.50, 0.60)},
            {"idx": 1, "status": "trained", "epochs": _epochs(0.40, 0.45)},
            {"idx": 2, "status": "running", "epochs": _epochs(0.99, 0.99)},   # not finished
            {"idx": 3, "status": "pruned", "epochs": _epochs(0.10)},          # not a fair baseline
            {"idx": 4, "status": "evaluated", "epochs": _epochs(None, None)}, # no metric
        ],
    }
    # current run is idx 2; only the two completed runs with a metric count
    assert sorted(_peer_bests_at(job, current_idx=2, epoch=2)) == [0.45, 0.60]
