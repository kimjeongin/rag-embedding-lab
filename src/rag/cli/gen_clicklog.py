"""`rag-gen-clicklog` — 노이즈 섞인 세션 클릭로그(리허설용)를 생성한다.

data-intranet/의 인트라넷 corpus를 전제로 한다 (doc_id = page-N). 산출물:

    clicklog.jsonl        세션 이벤트 (데이터 탭 '클릭로그' 가져오기에 그대로 붙여넣기 가능)
    clicklog-truth.jsonl  쿼리별 정답 (클리닝 품질 채점용 — 실로그에는 없는 것)

Env: INTRANET_DIR (기본 data-intranet), CLICKLOG_SESSIONS (기본 600), CLICKLOG_SEED.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from rag.dataset import write_jsonl
from rag.datagen.clicklog import clean, naive_pairs
from rag.datagen.clicklog_sim import SimConfig, simulate


def main() -> None:
    out = os.getenv("INTRANET_DIR", "data-intranet")
    config = SimConfig(n_sessions=int(os.getenv("CLICKLOG_SESSIONS", "600")))
    events, truth = simulate(int(os.getenv("CLICKLOG_SEED", "20260720")), config)

    write_jsonl(str(Path(out) / "clicklog.jsonl"), events)
    write_jsonl(
        str(Path(out) / "clicklog-truth.jsonl"),
        [{"query": q, "doc_id": d, "slice": s} for q, (d, s) in sorted(truth.items())],
    )

    slices = Counter(s for _, s in truth.values())
    cleaned = clean(events)
    print(f"[gen-clicklog] {out}/clicklog.jsonl — 이벤트 {len(events)} "
          f"(truth 쿼리 {len(truth)}: standard {slices['standard']} / jargon {slices['jargon']})")
    print(f"  naive 쌍 {len(naive_pairs(events))} vs cleaned 쌍 {len(cleaned.pairs)} · "
          f"리포트 {cleaned.report}")
