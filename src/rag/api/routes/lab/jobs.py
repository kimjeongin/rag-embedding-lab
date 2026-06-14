"""/api/jobs — create, observe and control server-owned training jobs.

The browser POSTs a job (one run or a sweep), gets an id back immediately, and from
then on just polls GET /api/jobs/{id} — closing the tab changes nothing. One job runs
at a time (one training device); a second POST while one is active is a 409. All the
actual work lives in ``rag.api.jobs``; these handlers map state to DTOs.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag.api import jobs
from rag.api.schemas.lab import (
    JobCreateRequest,
    JobsListResponse,
    JobState,
    JobSummary,
)

router = APIRouter()

_TERMINAL = ("trained", "evaluated", "failed", "skipped", "stopped", "interrupted", "pruned")


def _summary(job: dict) -> JobSummary:
    runs = job.get("runs", [])
    return JobSummary(
        id=job["id"],
        kind=job["kind"],
        status=job["status"],
        created_at=job["created_at"],
        n_runs=len(runs),
        n_finished=sum(1 for r in runs if r.get("status") in _TERMINAL),
        labels=[r.get("label") or f"run {r['idx']}" for r in runs[:4]],
    )


@router.post("/jobs", response_model=JobState)
async def create_job(req: JobCreateRequest) -> JobState:
    if jobs.active_job_id() is not None:
        raise HTTPException(
            status_code=409,
            detail="이미 실행 중인 잡이 있습니다 — 끝나기를 기다리거나 중단한 뒤 시작하세요",
        )
    job = jobs.create_job(
        [{"label": r.label, "config": r.config.model_dump()} for r in req.runs],
        auto_eval=req.auto_eval,
        keep_top_k=req.keep_top_k,
        prune=req.prune,
    )
    jobs.start_job(job["id"])
    return JobState(**job)


@router.get("/jobs", response_model=JobsListResponse)
def list_jobs() -> JobsListResponse:
    return JobsListResponse(
        jobs=[_summary(j) for j in jobs.list_jobs()],
        active=jobs.active_job_id(),
    )


@router.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str) -> JobState:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="해당 잡이 없습니다")
    return JobState(**job)


@router.post("/jobs/{job_id}/stop", response_model=JobState)
def stop_job(job_id: str) -> JobState:
    if not jobs.request_stop(job_id):
        raise HTTPException(status_code=409, detail="이 잡은 실행 중이 아닙니다")
    return JobState(**jobs.get_job(job_id))


@router.post("/jobs/{job_id}/skip", response_model=JobState)
def skip_run(job_id: str) -> JobState:
    if not jobs.request_skip(job_id):
        raise HTTPException(status_code=409, detail="이 잡은 실행 중이 아닙니다")
    return JobState(**jobs.get_job(job_id))


@router.delete("/jobs/{job_id}", response_model=JobsListResponse)
def delete_job(job_id: str) -> JobsListResponse:
    if not jobs.delete_job(job_id):
        raise HTTPException(status_code=409, detail="실행 중이거나 존재하지 않는 잡은 삭제할 수 없습니다")
    return JobsListResponse(jobs=[_summary(j) for j in jobs.list_jobs()], active=jobs.active_job_id())
