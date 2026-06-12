"""POST /api/eval — measure one model and record the run.

Thin HTTP wrapper over ``rag.evalflow`` (the same flow the job runner uses for
auto-eval): auto-detects the embedding dimension, ranks the eval corpus, computes
recall@k / MRR@10 / nDCG@10 with a bootstrap 95% CI, appends the result (fingerprint,
per-query scores, top-10 rankings) to the registry, and returns the metrics with the
prior best on the SAME eval set + split. ``split="final"`` runs the one-shot
confirmation on the held-out qrels. Upstream embedding failures surface as 502; an
eval set with no judged queries is a 422; a missing final split is a 409.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag.api.schemas.lab import EvalRequest, EvalResponse, RunRecord
from rag.evalflow import NoJudgedQueries, run_eval_flow

router = APIRouter()


@router.post("/eval", response_model=EvalResponse)
async def run_eval(req: EvalRequest) -> EvalResponse:
    try:
        result = await run_eval_flow(
            req.embedder,
            req.model,
            label=req.label,
            eval_dir=req.eval_dir,
            split=req.split,
            ollama_url=req.ollama_url,
            note=req.note,
        )
    except NoJudgedQueries as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # final split missing → actionable 409, not a 500
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface model/embedding failures as 502
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    return EvalResponse(
        model=result["model"],
        embed_dim=result["embed_dim"],
        metrics=result["metrics"],
        n_queries=result["n_queries"],
        ci95=result["ci95"],
        run=RunRecord(**result["run"]),
        prior_best=result["prior_best"],
        split=result["split"],
    )
