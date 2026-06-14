"""Cross-run median pruning for sweeps — pure decision logic, no I/O or globals.

While a run trains, compare its best-so-far validation metric against the runs that
already finished, AT THE SAME EPOCH, and stop it early if it trails the median (the
single biggest compute saving for an expensive sequential sweep — Optuna's MedianPruner
on the metric the runner already streams). Lives apart from the job runner
(``rag.api.jobs``) because it's pure and independently testable.

Metric-aware: a sweep monitors either val nDCG@10 (higher is better) or val loss (lower
is better). Pruning judges on whichever the run monitors, so it never fights the
in-process early stopping that keeps the best epoch on the SAME metric.
"""
from __future__ import annotations

import statistics

WARMUP = 3        # need ≥ this many completed runs before a median is trustworthy
MIN_EPOCH = 2     # never prune before a run has had this many epochs (early curves are noisy)
MARGIN = 1e-4     # must trail by more than this — never prune on a near-tie


def best_metric_at(epochs: list[dict], epoch: int, metric: str) -> float | None:
    """A run's best monitored value through ``epoch`` — max nDCG, or min eval_loss (the
    value its early stopping would keep). None when no epoch up to here reported it."""
    key = "ndcg" if metric == "ndcg" else "eval_loss"
    seen = [e[key] for e in epochs if e.get(key) is not None and e.get("epoch", 0) <= epoch]
    if not seen:
        return None
    return max(seen) if metric == "ndcg" else min(seen)


def should_prune(current: float | None, peer_bests: list[float], epoch: int, metric: str) -> bool:
    """True when a run at ``epoch`` trails the completed peers' median on ``metric``.
    Higher-is-better for nDCG, lower-is-better for loss; a MARGIN keeps near-ties alive."""
    if epoch < MIN_EPOCH or current is None or len(peer_bests) < WARMUP:
        return False
    median = statistics.median(peer_bests)
    if metric == "ndcg":
        return current < median - MARGIN
    return current > median + MARGIN   # loss: trailing means HIGHER than the median


def peer_bests_at(runs: list[dict], current_idx: int, epoch: int, metric: str) -> list[float]:
    """best-so-far ``metric`` at ``epoch`` for every COMPLETED run other than this one.
    Completed-only: a run that was itself pruned/failed isn't a fair baseline to judge by.
    """
    out = []
    for run in runs:
        if run["idx"] == current_idx or run.get("status") not in ("trained", "evaluated"):
            continue
        best = best_metric_at(run.get("epochs") or [], epoch, metric)
        if best is not None:
            out.append(best)
    return out
