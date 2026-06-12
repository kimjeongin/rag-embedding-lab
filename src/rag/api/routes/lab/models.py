"""/api/models — pick, inspect, delete and hand off saved models.

GET /models is the picker the Evaluate/Train screens use (Ollama tags or saved ST
dirs). GET /models/detail joins each saved model with its recipe (train_meta.json),
disk size and eval records — the models page + Compare's hyperparameter columns.
DELETE removes one model dir (guarded to outputs/; refused while a job is running).
POST /models/handoff packages the chosen winner for the serving team (the lab's
finish line — production swaps the dense model inside its own pipeline).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from rag import lab, modelstore
from rag.api import jobs
from rag.api.deps import get_settings
from rag.api.schemas.lab import (
    DeleteModelResponse,
    Embedder,
    HandoffRequest,
    HandoffResponse,
    ModelsDetailResponse,
    ModelsResponse,
)
from rag.config import Settings

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def models(
    embedder: Embedder = "ollama",
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    choices = lab.list_models(embedder, settings.ollama_url)
    return ModelsResponse(
        embedder=embedder,
        models=choices,
        default=lab.default_model(embedder, choices),
    )


@router.get("/models/detail", response_model=ModelsDetailResponse)
def models_detail() -> ModelsDetailResponse:
    return ModelsDetailResponse(**modelstore.list_detail(lab.list_st_models()))


@router.delete("/models", response_model=DeleteModelResponse)
def delete_model(path: str) -> DeleteModelResponse:
    if jobs.active_job_id() is not None:
        raise HTTPException(status_code=409, detail="잡 실행 중에는 모델을 삭제할 수 없습니다")
    try:
        modelstore.delete_model(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DeleteModelResponse(deleted=path, **modelstore.list_detail(lab.list_st_models()))


@router.post("/models/handoff", response_model=HandoffResponse)
async def handoff(req: HandoffRequest) -> HandoffResponse:
    if not lab.training_ready():
        raise HTTPException(
            status_code=503,
            detail="핸드오프 패키지 생성에는 학습 스택이 필요합니다 — `uv sync --group training`",
        )
    try:
        # blocking torch encode → off the event loop
        result = await asyncio.to_thread(modelstore.build_handoff, req.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface encode/load failures as 502
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    return HandoffResponse(path=req.path, markdown=result["markdown"], handoff=result["handoff"])
