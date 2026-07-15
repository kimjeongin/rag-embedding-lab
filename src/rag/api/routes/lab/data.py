"""/api/data/* — inspect and (re)generate the lab's datasets.

Reads: an overview (what exists + where it's consumed), plus small previews of the
training pairs and the eval corpus. Writes: regenerate training pairs (toy split or an
LLM-synthesised set) and the BEIR-format eval set. All of it delegates to
``rag.datagen`` / ``rag.dataset`` / ``rag.evaluation`` — the route just maps to DTOs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from rag import lab, serving
from rag.api import labelrank
from rag.api.deps import get_embedder, get_settings, get_store
from rag.api.schemas.lab import (
    CorpusDoc,
    CorpusResponse,
    CrawlRequest,
    DataOverviewResponse,
    EvalInfo,
    FileCount,
    GenEvalRequest,
    GenEvalResponse,
    GenPairsRequest,
    GenPairsResponse,
    ImportPairsRequest,
    ImportPairsResponse,
    LabelCommitRequest,
    LabelCommitResponse,
    LabelDoc,
    LabelSearchRequest,
    LabelSearchResponse,
    PairItem,
    PairsResponse,
)
from rag.api.sse import sse_event
from rag.config import Settings
from rag.core.errors import VectorStoreError
from rag.datagen import ingest
from rag.datagen.dummy import generate_dataset
from rag.datagen.eval_corpus import generate as generate_eval_set
from rag.datagen.eval_corpus import split_qrels as split_eval_qrels
from rag.dataset import dataset_paths, load_jsonl, write_jsonl
from rag.evaluation.beir import (
    DEV_SPLIT,
    FINAL_SPLIT,
    eval_dir_from_env,
    load_corpus,
    load_qrels,
    prune_qrels_splits,
    resolve_split,
    write_beir_dataset,
    write_qrels,
)

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


def _train_has_negatives(train_file: str) -> bool:
    """True when the dataset carries mined hard negatives (TripletLoss's requirement).
    Any record having them is enough — training pads short records from the others'
    docs (rag.training.data.pad_negatives)."""
    if not Path(train_file).exists():
        return False
    records = list(load_jsonl(train_file))
    return any(r.get("negatives") for r in records)


# ── reads ───────────────────────────────────────────────────────────────────────
@router.get("/data/overview", response_model=DataOverviewResponse)
def data_overview() -> DataOverviewResponse:
    train_file, test_file = dataset_paths()
    return DataOverviewResponse(
        train=FileCount(file=train_file, count=lab.count_lines(train_file)),
        test=FileCount(file=test_file, count=lab.count_lines(test_file)),
        train_has_negatives=_train_has_negatives(train_file),
        eval=EvalInfo(**lab.eval_overview(eval_dir_from_env())),
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
                req.corpus_file, req.gen_model or "", req.n_queries, req.hard_negatives,
                Settings.from_env(), round_trip_k=req.round_trip_k, neg_margin=req.neg_margin,
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
                req.corpus_file, req.gen_model or "", req.n_queries, req.hard_negatives,
                Settings.from_env(), round_trip_k=req.round_trip_k, neg_margin=req.neg_margin,
            ):
                if ev["event"] != "done":
                    yield sse_event(ev["event"], {k: v for k, v in ev.items() if k != "event"})
                    continue
                train, test = ev["train"], ev["test"]
                write_jsonl(train_file, train)
                write_jsonl(test_file, test)
                _, preview = _pair_items(train_file, 8, with_content=False)
                yield sse_event("done", {
                    "message": f"학습쌍 저장: {train_file} ({len(train)}) + {test_file} ({len(test)})",
                    "train": {"file": train_file, "count": len(train)},
                    "test": {"file": test_file, "count": len(test)},
                    "preview": [p.model_dump() for p in preview],
                })
        except Exception as exc:  # noqa: BLE001 — surface upstream (Ollama) failures into the stream
            yield sse_event("error", {
                "detail": f"{type(exc).__name__}: {exc} (Ollama 실행 중인가요? '{req.gen_model}' 모델을 받으셨나요?)"
            })

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/data/crawl/stream")
async def crawl_corpus_stream(req: CrawlRequest) -> StreamingResponse:
    """Crawl a public site into the corpus file, streamed over SSE (start/page/done/error).

    The PoC corpus source — same engine as the ``rag-crawl`` CLI. The ``done`` event
    fires AFTER the corpus file is written, so a client that sees it can refetch
    counts immediately.
    """
    async def _events():
        from rag.datagen.crawl import crawl_stream

        try:
            async for ev in crawl_stream(req.url, max_pages=req.max_pages):
                if ev["event"] != "done":
                    yield sse_event(ev["event"], {k: v for k, v in ev.items() if k != "event"})
                    continue
                pages = ev["pages"]
                if not pages:
                    yield sse_event("error", {
                        "detail": "수집된 페이지가 없습니다 — URL과 robots.txt를 확인하세요"
                    })
                    return
                write_jsonl(req.corpus_file, pages)
                yield sse_event("done", {
                    "message": f"corpus 저장: {req.corpus_file} ({len(pages)} pages)",
                    "file": req.corpus_file,
                    "count": len(pages),
                    "fetched": ev["fetched"],
                    "skipped": ev["skipped"],
                })
        except Exception as exc:  # noqa: BLE001 — surface crawl failures into the stream
            yield sse_event("error", {"detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── real data in: import + labeling ──────────────────────────────────────────────
def _append_train_pairs(pairs: list[dict]) -> int:
    """Append deduped pairs to train.jsonl; returns how many were actually new."""
    train_file, _ = dataset_paths()
    existing = list(load_jsonl(train_file)) if Path(train_file).exists() else []
    fresh = ingest.dedupe_pairs(existing, pairs)
    if fresh:
        write_jsonl(train_file, existing + fresh)
    return len(fresh)


def _append_qrels(eval_dir: str, new_queries: list[dict], rows: list[tuple[str, str, int]]) -> int:
    """Merge new judgments into the ACTIVE tuning split (+ queries.jsonl). Changes the
    eval set's fingerprint — prior runs stop being comparable, by design."""
    split = resolve_split(eval_dir)
    merged: dict[tuple[str, str], int] = {}
    for query_id, judgments in load_qrels(eval_dir, split).items():
        for doc_id, score in judgments.items():
            merged[(query_id, doc_id)] = int(score)
    added = 0
    for query_id, doc_id, score in rows:
        if (query_id, doc_id) not in merged:
            merged[(query_id, doc_id)] = score
            added += 1
    write_qrels(eval_dir, [(q, d, s) for (q, d), s in merged.items()], split)

    queries_path = Path(eval_dir) / "queries.jsonl"
    existing_queries = list(load_jsonl(str(queries_path))) if queries_path.exists() else []
    write_jsonl(str(queries_path), existing_queries + new_queries)
    return added


@router.post("/data/import", response_model=ImportPairsResponse)
def import_pairs(req: ImportPairsRequest) -> ImportPairsResponse:
    records, errors = ingest.parse_records(req.content)
    if not records:
        raise HTTPException(status_code=400, detail="; ".join(errors) or "읽을 수 있는 레코드가 없습니다")

    eval_dir = eval_dir_from_env()
    corpus_path = Path(eval_dir) / "corpus.jsonl"
    corpus = load_corpus(eval_dir) if corpus_path.exists() else {}
    skipped = list(errors)
    added_train = added_qrels = 0
    fingerprint_changed = False

    if req.target in ("train", "both"):
        pairs, skip = ingest.to_train_pairs(records, corpus)
        skipped += skip
        added_train = _append_train_pairs(pairs)

    if req.target in ("qrels", "both"):
        if not corpus:
            skipped.append("평가 corpus가 없어 qrels는 건너뛰었습니다 — 평가셋을 먼저 만드세요")
        else:
            queries_path = Path(eval_dir) / "queries.jsonl"
            taken = {str(r["_id"]) for r in load_jsonl(str(queries_path))} if queries_path.exists() else set()
            new_queries, rows, skip = ingest.to_qrels(records, corpus, taken)
            skipped += skip
            if rows:
                added_qrels = _append_qrels(eval_dir, new_queries, rows)
                fingerprint_changed = added_qrels > 0

    message = f"가져오기 완료 — 학습쌍 +{added_train}, qrels +{added_qrels}"
    if fingerprint_changed:
        message += " · 평가셋 내용이 바뀌어 이전 런과는 비교되지 않습니다 (새 fingerprint)"
    return ImportPairsResponse(
        parsed=len(records),
        added_train=added_train,
        added_qrels=added_qrels,
        skipped=skipped,
        fingerprint_changed=fingerprint_changed,
        message=message,
    )


async def _rank_via_serving_index(
    settings: Settings, embedder, store, corpus: dict, query: str, top_n: int = 10
) -> list[str] | None:
    """Fast path: the live Qdrant index already holds this model's document vectors
    (a ~30min embed pass on a laptop) — reuse them instead of re-embedding the
    corpus. Returns None whenever the index can't answer for THIS eval corpus
    (unreachable / no index / different model or dim / contents drifted), so the
    caller falls back to the in-process matrix."""
    status = await asyncio.to_thread(serving.index_status, settings, store)
    if not (status["reachable"] and status["collection"]):
        return None
    if not (status["dim_matches"] and status["model_matches"]):
        return None
    try:
        # over-fetch: serving corpus may hold docs outside the eval corpus
        result = await serving.search(settings, embedder, store, query, top_k=max(top_n * 3, 30))
    except VectorStoreError:
        return None
    return labelrank.map_hits_to_corpus_ids(result["hits"], corpus, top_n) or None


@router.post("/data/label/search", response_model=LabelSearchResponse)
async def label_search(
    req: LabelSearchRequest,
    request: Request,
    process: Settings = Depends(get_settings),
    store=Depends(get_store),
) -> LabelSearchResponse:
    """The judging loop's first half: rank the corpus for one query with the chosen
    model, so a human can click which results are actually relevant.

    When the chosen stack IS the serving embedder and the live index matches, the
    ranking reuses the index's vectors (ms). Otherwise the corpus matrix is built
    in-process and cached per (backend, model, corpus content) in ``labelrank`` —
    the first judgment pays the full embed pass, the rest embed one query each."""
    eval_dir = eval_dir_from_env()
    if not (Path(eval_dir) / "corpus.jsonl").exists():
        raise HTTPException(status_code=400, detail="평가 corpus가 없습니다 — 평가셋을 먼저 만드세요")
    corpus = load_corpus(eval_dir)

    try:
        dim = lab.infer_dim(req.embedder, req.model, process.ollama_url)
        eval_settings = lab.build_eval_settings(req.embedder, req.model, dim, process.ollama_url)
        ranked = None
        shared = None
        if (
            eval_settings.embedder == process.embedder
            and eval_settings.active_model == process.active_model
            and eval_settings.embed_dim == process.embed_dim
        ):
            # the judging default IS the serving embedder — borrow the process
            # instance instead of loading a second copy of the same ~GB weights
            shared = await get_embedder(request)
            ranked = await _rank_via_serving_index(process, shared, store, corpus, req.query)
        if ranked is None:
            ranked = await labelrank.rank(
                eval_settings, corpus, labelrank.corpus_fingerprint(eval_dir), req.query,
                shared=shared,
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface embedding failures as 502
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    results = []
    for doc_id in ranked:
        doc = corpus.get(doc_id) or {}
        text = doc.get("text") or ""
        results.append(LabelDoc(id=doc_id, title=doc.get("title"), text=text[:200] + ("…" if len(text) > 200 else "")))
    return LabelSearchResponse(query=req.query, results=results)


@router.post("/data/label/commit", response_model=LabelCommitResponse)
def label_commit(req: LabelCommitRequest) -> LabelCommitResponse:
    """The judging loop's second half: the clicked docs become qrels (and, by default,
    training pairs too) — every judged query makes the eval set more real."""
    eval_dir = eval_dir_from_env()
    if not (Path(eval_dir) / "corpus.jsonl").exists():
        raise HTTPException(status_code=400, detail="평가 corpus가 없습니다 — 평가셋을 먼저 만드세요")
    corpus = load_corpus(eval_dir)
    missing = [d for d in req.doc_ids if d not in corpus]
    if missing:
        raise HTTPException(status_code=400, detail=f"corpus에 없는 문서: {', '.join(missing)}")

    queries_path = Path(eval_dir) / "queries.jsonl"
    taken = {str(r["_id"]) for r in load_jsonl(str(queries_path))} if queries_path.exists() else set()
    records = [{"query": req.query, "doc_id": doc_id} for doc_id in req.doc_ids]
    new_queries, rows, _ = ingest.to_qrels(records, corpus, taken)
    added_qrels = _append_qrels(eval_dir, new_queries, rows)

    added_train = 0
    if req.also_train:
        pairs, _ = ingest.to_train_pairs(records, corpus)
        added_train = _append_train_pairs(pairs)

    query_id = new_queries[0]["_id"] if new_queries else rows[0][0]
    return LabelCommitResponse(
        query_id=query_id,
        added_qrels=added_qrels,
        added_train=added_train,
        message=(
            f"'{req.query}' → {query_id} (qrels +{added_qrels}, 학습쌍 +{added_train}) · "
            "평가셋 내용이 바뀌어 이전 런과는 비교되지 않습니다"
        ),
    )


@router.post("/data/eval", response_model=GenEvalResponse)
def gen_eval(req: GenEvalRequest) -> GenEvalResponse:
    eval_dir = eval_dir_from_env()
    note = ""
    if req.source == "corpus":
        # The real corpus is the haystack; the held-out test split is the query set.
        from rag.datagen.eval_from_corpus import load_and_build

        _, test_file = dataset_paths()
        try:
            corpus, queries, qrels, skipped = load_and_build(req.corpus_file, test_file)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if skipped:
            note = f" · 매칭 안 된 test 쌍 {skipped}건 제외"
    else:
        corpus, queries, qrels = generate_eval_set(n_distractors=req.n_distractors)
    # dev (tuning) / final (held-out one-shot confirmation) — see rag.evaluation.beir
    dev_rows, final_rows = split_eval_qrels(qrels)
    write_beir_dataset(eval_dir, corpus, queries, dev_rows, split=DEV_SPLIT)
    write_qrels(eval_dir, final_rows, FINAL_SPLIT)
    # a split file from the previous generation would point at ids that no longer exist
    prune_qrels_splits(eval_dir, keep=(DEV_SPLIT, FINAL_SPLIT))
    _, preview = _corpus_items(eval_dir, 8, truncate=120)
    return GenEvalResponse(
        message=(
            f"평가셋 저장 ({req.source}): {eval_dir} ({len(corpus)} docs · {len(queries)} queries · "
            f"qrels dev {len(dev_rows)} + final {len(final_rows)}){note}"
        ),
        dir=eval_dir,
        corpus=len(corpus),
        queries=len(queries),
        qrels=len(qrels),
        preview=preview,
    )
