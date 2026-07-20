"""`rag-gen-clicklog` — 노이즈 섞인 세션 클릭로그(리허설용)를 생성한다.

data-intranet/의 인트라넷 corpus를 전제로 한다 (doc_id = page-N). 산출물:

    clicklog.jsonl        세션 이벤트 (데이터 탭 '클릭로그' 가져오기에 그대로 붙여넣기 가능)
    clicklog-truth.jsonl  쿼리별 정답 (클리닝 품질 채점용 — 실로그에는 없는 것)

학습 대조 실험용으로 로그에서 뽑은 두 학습 파일도 함께 쓴다 (corpus로 해석):

    train-clicklog-clean.jsonl   클리닝 계층 통과 (전이 쌍 + hard negative 포함)
    train-clicklog-naive.jsonl   현행 방식 (클릭 1건 = 쌍 1건, 중복 제거만)

Env: INTRANET_DIR (기본 data-intranet), CLICKLOG_SESSIONS (기본 600), CLICKLOG_SEED.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from rag.dataset import load_jsonl, write_jsonl
from rag.datagen.clicklog import clean, naive_pairs
from rag.datagen.clicklog_sim import SimConfig, simulate


def _resolve(pairs: list[dict], corpus: dict[str, dict]) -> list[dict]:
    """doc_id 쌍 → 학습 포맷 (positive/negatives를 corpus 문서로 해석, 중복 제거)."""
    out, seen = [], set()
    for pair in pairs:
        doc = corpus.get(pair["doc_id"])
        key = (pair["query"].lower(), pair["doc_id"])
        if doc is None or key in seen:
            continue
        seen.add(key)
        record = {
            "query": pair["query"],
            "positive": {"title": doc.get("title"), "content": doc["text"]},
        }
        negatives = [
            {"title": corpus[d].get("title"), "content": corpus[d]["text"]}
            for d in pair.get("negatives") or [] if d in corpus
        ]
        if negatives:
            record["negatives"] = negatives
        out.append(record)
    return out


def main() -> None:
    out = os.getenv("INTRANET_DIR", "data-intranet")
    config = SimConfig(n_sessions=int(os.getenv("CLICKLOG_SESSIONS", "600")))
    events, truth = simulate(int(os.getenv("CLICKLOG_SEED", "20260720")), config)

    write_jsonl(str(Path(out) / "clicklog.jsonl"), events)
    write_jsonl(
        str(Path(out) / "clicklog-truth.jsonl"),
        [{"query": q, "doc_id": d, "slice": s} for q, (d, s) in sorted(truth.items())],
    )

    corpus = {
        str(r["_id"]): {"title": r.get("title"), "text": r["text"]}
        for r in load_jsonl(str(Path(out) / "eval" / "corpus.jsonl"))
    }
    cleaned = clean(events)
    clean_rows = _resolve(cleaned.pairs, corpus)
    naive_rows = _resolve(naive_pairs(events), corpus)
    write_jsonl(str(Path(out) / "train-clicklog-clean.jsonl"), clean_rows)
    write_jsonl(str(Path(out) / "train-clicklog-naive.jsonl"), naive_rows)

    slices = Counter(s for _, s in truth.values())
    print(f"[gen-clicklog] {out}/clicklog.jsonl — 이벤트 {len(events)} "
          f"(truth 쿼리 {len(truth)}: standard {slices['standard']} / jargon {slices['jargon']})")
    print(f"  train-clicklog-clean {len(clean_rows)} 쌍 (negatives 포함 "
          f"{sum(1 for r in clean_rows if r.get('negatives'))}) vs naive {len(naive_rows)} 쌍 · "
          f"리포트 {cleaned.report}")
