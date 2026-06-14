"""Cross-run median-pruning decision — the pure helpers in rag.api.pruning.

A sweep run is pruned when its best-so-far validation metric trails the median of the
runs that already finished, judged at the same epoch. Metric-aware: nDCG is
higher-is-better, val loss is lower-is-better. Held off until a couple of epochs in and
until enough peers exist. No torch / subprocess needed — plain functions over the epoch
points the runner already streams.
"""
from rag.api.pruning import (
    MARGIN,
    MIN_EPOCH,
    WARMUP,
    best_metric_at,
    peer_bests_at,
    should_prune,
)


def _ndcg(*vals):
    return [{"epoch": i + 1, "ndcg": v, "eval_loss": None} for i, v in enumerate(vals)]


def _loss(*vals):
    return [{"epoch": i + 1, "ndcg": None, "eval_loss": v} for i, v in enumerate(vals)]


def test_best_metric_at_running_best_through_an_epoch():
    nd = _ndcg(0.50, 0.42, 0.61, 0.58)
    assert best_metric_at(nd, 2, "ndcg") == 0.50      # max(0.50, 0.42)
    assert best_metric_at(nd, 4, "ndcg") == 0.61
    ls = _loss(0.90, 0.70, 0.80)
    assert best_metric_at(ls, 2, "loss") == 0.70      # min(0.90, 0.70) — lower is better
    assert best_metric_at(_ndcg(None, None), 2, "ndcg") is None
    assert best_metric_at([], 3, "ndcg") is None


def test_should_prune_ndcg_only_when_trailing_the_peer_median():
    peers = [0.50, 0.55, 0.60]                        # median 0.55, higher is better
    assert should_prune(0.40, peers, epoch=3, metric="ndcg") is True
    assert should_prune(0.58, peers, epoch=3, metric="ndcg") is False
    assert should_prune(0.55, peers, epoch=3, metric="ndcg") is False        # tie at median → keep
    assert should_prune(0.55 - MARGIN / 2, peers, epoch=3, metric="ndcg") is False  # within margin


def test_should_prune_loss_flips_direction():
    peers = [0.30, 0.35, 0.40]                        # median 0.35, LOWER is better
    assert should_prune(0.50, peers, epoch=3, metric="loss") is True   # higher loss = trailing
    assert should_prune(0.30, peers, epoch=3, metric="loss") is False  # lower loss = leading


def test_should_prune_holds_off_early_and_without_enough_peers():
    peers = [0.50, 0.55, 0.60]
    assert should_prune(0.10, peers, epoch=MIN_EPOCH - 1, metric="ndcg") is False  # too early
    too_few = [0.55] * (WARMUP - 1)
    assert should_prune(0.10, too_few, epoch=5, metric="ndcg") is False            # < WARMUP peers
    assert should_prune(None, peers, epoch=5, metric="ndcg") is False              # no metric yet


def test_peer_bests_skips_self_and_unfinished_runs():
    runs = [
        {"idx": 0, "status": "evaluated", "epochs": _ndcg(0.50, 0.60)},
        {"idx": 1, "status": "trained", "epochs": _ndcg(0.40, 0.45)},
        {"idx": 2, "status": "running", "epochs": _ndcg(0.99, 0.99)},   # not finished
        {"idx": 3, "status": "pruned", "epochs": _ndcg(0.10)},          # not a fair baseline
        {"idx": 4, "status": "evaluated", "epochs": _ndcg(None, None)}, # no metric
    ]
    # current run is idx 2; only the two completed runs with a metric count
    assert sorted(peer_bests_at(runs, current_idx=2, epoch=2, metric="ndcg")) == [0.45, 0.60]
