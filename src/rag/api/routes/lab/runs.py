"""/api/runs — the eval-run registry behind the Overview leaderboard + Compare table.

List is newest-first and carries the best-per-metric map (so the UI can highlight the
winner and show Δ-vs-best) plus the metric display order. Delete removes one run by id.
Both are thin wrappers over ``rag.runs`` (stdlib JSONL store).
"""
from __future__ import annotations

from fastapi import APIRouter

from rag import runs as registry
from rag.api.schemas.lab import DeleteRunResponse, RunRecord, RunsResponse

router = APIRouter()


@router.get("/runs", response_model=RunsResponse)
def list_runs() -> RunsResponse:
    return RunsResponse(
        runs=[RunRecord(**r) for r in registry.load_runs()],
        best=registry.best_per_metric(),
        metric_keys=list(registry.METRIC_KEYS),
    )


@router.delete("/runs/{run_id}", response_model=DeleteRunResponse)
def delete_run(run_id: str) -> DeleteRunResponse:
    remaining = registry.delete_run(run_id)
    return DeleteRunResponse(deleted=run_id, remaining=remaining)
