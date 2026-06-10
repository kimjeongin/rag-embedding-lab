"""/api/runs — the eval-run registry behind the Overview leaderboard + Compare table.

List is newest-first and carries the best-per-metric map (so the UI can highlight the
winner and show Δ-vs-best) plus the metric display order. ``best`` and the returned
``current_fingerprint`` are scoped to the eval set currently bound to the process —
runs measured on other (or regenerated) sets aren't comparable, and the UI uses the
fingerprint to keep them out of rankings. Delete removes one run by id. Thin wrappers
over ``rag.runs`` (stdlib JSONL store).
"""
from __future__ import annotations

from fastapi import APIRouter

from rag import runs as registry
from rag.api.schemas.lab import DeleteRunResponse, RunRecord, RunsResponse
from rag.evaluation.beir import eval_dir_from_env, eval_set_fingerprint

router = APIRouter()


@router.get("/runs", response_model=RunsResponse)
def list_runs() -> RunsResponse:
    current = eval_set_fingerprint(eval_dir_from_env())
    return RunsResponse(
        runs=[RunRecord(**r) for r in registry.load_runs()],
        best=registry.best_per_metric(fingerprint=current),
        current_fingerprint=current,
        metric_keys=list(registry.METRIC_KEYS),
    )


@router.delete("/runs/{run_id}", response_model=DeleteRunResponse)
def delete_run(run_id: str) -> DeleteRunResponse:
    remaining = registry.delete_run(run_id)
    return DeleteRunResponse(deleted=run_id, remaining=remaining)
