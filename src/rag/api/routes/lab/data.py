"""/api/data/* — inspect and (re)generate the lab's datasets.

Reads: an overview (what exists + where it's consumed), plus small previews of the
training pairs and the eval corpus. Writes: regenerate training pairs (toy split or an
LLM-synthesised set) and the BEIR-format eval set. All of it delegates to
``rag.datagen`` / ``rag.dataset`` / ``rag.evaluation`` — the route just maps to DTOs.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag import lab
from rag.api.schemas.lab import (
    CorpusDoc,
    CorpusResponse,
    DataOverviewResponse,
    EvalInfo,
    FileCount,
    GenEvalRequest,
    GenEvalResponse,
    GenPairsRequest,
    GenPairsResponse,
    PairItem,
    PairsResponse,
)
from rag.config import Settings
from rag.datagen.dummy import generate_dataset
from rag.datagen.eval_corpus import generate as generate_eval_set
from rag.dataset import dataset_paths, load_jsonl, write_jsonl
from rag.evaluation.beir import eval_dir_from_env, write_beir_dataset

router = APIRouter()


# ── preview helpers (total, items) ──────────────────────────────────────────────
def _pair_items(path: str, limit: int | None, with_content: bool) -> tuple[int, list[PairItem]]:
    records = list(load_jsonl(path)) if Path(path).exists() else []
    total = len(records)
    chosen = records[:limit] if limit else records
    items = [
        PairItem(
            query=r.get("query"),
            title=(r.get("positive") or {}).get("title"),
            content=((r.get("positive") or {}).get("content", "") if with_content else None),
        )
        for r in chosen
    ]
    return total, items


def _corpus_items(eval_dir: str, limit: int | None, truncate: int | None) -> tuple[int, list[CorpusDoc]]:
    path = f"{eval_dir}/corpus.jsonl"
    records = list(load_jsonl(path)) if Path(path).exists() else []
    total = len(records)
    chosen = records[:limit] if limit else records
    items = []
    for r in chosen:
        text = r.get("text", "")
        if truncate and len(text) > truncate:
            text = text[:truncate] + "…"
        items.append(CorpusDoc(id=r.get("_id"), title=r.get("title"), text=text))
    return total, items


# ── reads ───────────────────────────────────────────────────────────────────────
@router.get("/data/overview", response_model=DataOverviewResponse)
def data_overview() -> DataOverviewResponse:
    train_file, test_file = dataset_paths()
    eval_dir = eval_dir_from_env()
    return DataOverviewResponse(
        train=FileCount(file=train_file, count=lab.count_lines(train_file)),
        test=FileCount(file=test_file, count=lab.count_lines(test_file)),
        eval=EvalInfo(
            dir=eval_dir,
            is_sample=lab.is_sample_eval(eval_dir),
            corpus=lab.count_lines(f"{eval_dir}/corpus.jsonl"),
            queries=lab.count_lines(f"{eval_dir}/queries.jsonl"),
        ),
    )


@router.get("/data/pairs", response_model=PairsResponse)
def data_pairs(limit: int | None = 8, content: bool = False) -> PairsResponse:
    train_file, _ = dataset_paths()
    total, items = _pair_items(train_file, limit, content)
    return PairsResponse(file=train_file, total=total, items=items)


@router.get("/data/corpus", response_model=CorpusResponse)
def data_corpus(limit: int | None = 8, truncate: int | None = None) -> CorpusResponse:
    eval_dir = eval_dir_from_env()
    total, items = _corpus_items(eval_dir, limit, truncate)
    return CorpusResponse(dir=eval_dir, total=total, items=items)


# ── writes ──────────────────────────────────────────────────────────────────────
@router.post("/data/pairs", response_model=GenPairsResponse)
async def gen_pairs(req: GenPairsRequest) -> GenPairsResponse:
    train_file, test_file = dataset_paths()
    if req.method == "toy":
        train, test = generate_dataset()
    else:
        if not req.corpus_file:
            raise HTTPException(status_code=400, detail="synthetic 생성에는 corpus_file이 필요합니다")
        from rag.datagen.synthetic import generate

        try:
            train, test = await generate(
                req.corpus_file, req.gen_model or "", req.n_queries, req.hard_negatives, Settings.from_env()
            )
        except Exception as exc:  # noqa: BLE001 — surface upstream (Ollama) failures as 502
            raise HTTPException(
                status_code=502,
                detail=f"{type(exc).__name__}: {exc} (Ollama 실행 중인가요? '{req.gen_model}' 모델을 받으셨나요?)",
            ) from exc

    write_jsonl(train_file, train)
    write_jsonl(test_file, test)
    _, preview = _pair_items(train_file, 8, with_content=False)
    return GenPairsResponse(
        message=f"학습쌍 저장: {train_file} ({len(train)}) + {test_file} ({len(test)})",
        train=FileCount(file=train_file, count=len(train)),
        test=FileCount(file=test_file, count=len(test)),
        preview=preview,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/data/pairs/stream")
async def gen_pairs_stream(req: GenPairsRequest) -> StreamingResponse:
    """Synthetic training-pair generation, streamed over SSE (start/doc/mining/done/error).

    Same result as ``POST /data/pairs`` with ``method=synthetic``, but emits per-document
    progress so the UI isn't a black box during the (LLM-bound) run.
    """
    if not req.corpus_file:
        raise HTTPException(status_code=400, detail="synthetic 생성에는 corpus_file이 필요합니다")

    train_file, test_file = dataset_paths()

    async def _events():
        from rag.datagen.synthetic import generate_stream

        try:
            async for ev in generate_stream(
                req.corpus_file, req.gen_model or "", req.n_queries, req.hard_negatives, Settings.from_env()
            ):
                if ev["event"] != "done":
                    yield _sse(ev["event"], {k: v for k, v in ev.items() if k != "event"})
                    continue
                train, test = ev["train"], ev["test"]
                write_jsonl(train_file, train)
                write_jsonl(test_file, test)
                _, preview = _pair_items(train_file, 8, with_content=False)
                yield _sse("done", {
                    "message": f"학습쌍 저장: {train_file} ({len(train)}) + {test_file} ({len(test)})",
                    "train": {"file": train_file, "count": len(train)},
                    "test": {"file": test_file, "count": len(test)},
                    "preview": [p.model_dump() for p in preview],
                })
        except Exception as exc:  # noqa: BLE001 — surface upstream (Ollama) failures into the stream
            yield _sse("error", {
                "detail": f"{type(exc).__name__}: {exc} (Ollama 실행 중인가요? '{req.gen_model}' 모델을 받으셨나요?)"
            })

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/data/eval", response_model=GenEvalResponse)
def gen_eval(req: GenEvalRequest) -> GenEvalResponse:
    eval_dir = eval_dir_from_env()
    corpus, queries, qrels = generate_eval_set(n_distractors=req.n_distractors)
    write_beir_dataset(eval_dir, corpus, queries, qrels)
    _, preview = _corpus_items(eval_dir, 8, truncate=120)
    return GenEvalResponse(
        message=f"평가셋 저장: {eval_dir} ({len(corpus)} docs · {len(queries)} queries · {len(qrels)} qrels)",
        dir=eval_dir,
        corpus=len(corpus),
        queries=len(queries),
        qrels=len(qrels),
        preview=preview,
    )
