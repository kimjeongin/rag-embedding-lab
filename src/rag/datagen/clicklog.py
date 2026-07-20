"""세션 클릭로그 → 학습쌍 — 노이즈를 아는 만큼만 믿는 변환 계층.

실제 검색 클릭로그는 (쿼리, 클릭 문서)를 그대로 라벨로 쓰기엔 지저분하다:

  - **포지션 바이어스**: 1등이라서 클릭했지 정답이라서가 아니다.
  - **오클릭/바운스**: 잘못 눌렀다 바로 나온 클릭 (dwell 몇 초).
  - **재검색**: 첫 쿼리가 실패해 고쳐 검색한 세션 — 첫 쿼리 자체가 "엔진이 못 푸는
    표현"이라는 신호이고, 세션의 최종 만족 문서가 그 쿼리의 정답이다. 사내
    은어처럼 현행 엔진이 못 잡는 표현의 supervision은 대부분 여기서 나온다.
  - **PII**: 사번·이메일·전화번호가 쿼리에 섞여 들어온다 — 학습 데이터로 유출 금지.

이 모듈은 그 노이즈를 규칙별로 처리하고 **무엇을 왜 버렸는지 카운트로 보고**한다.
순수 변환(파일/네트워크 IO 없음) — 시뮬레이션 로그(rehearsal)와 실로그가 같은
코드를 통과한다.

입력 이벤트 (JSONL 한 줄 = 검색 1회):

    {"session": "s-7", "query": "돌핀 정산 어디서",
     "results": ["page-12", "page-3", ...],              # 노출 순서 (선택)
     "clicks": [{"doc_id": "page-3", "rank": 3, "dwell_sec": 4}, ...]}

세션 내 이벤트 순서는 파일 순서를 따른다 (로그는 시간순 export가 보통이다).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 쿼리에 개인정보가 섞인 이벤트는 학습 데이터로 내보내지 않는다 (마스킹은 쿼리
# 의미를 바꿔 검색 supervision으로서도 어긋난다 — 통째로 버리고 카운트).
_PII_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),              # 이메일
    re.compile(r"0\d{1,2}[ -]?\d{3,4}[ -]?\d{4}"),       # 전화번호
    re.compile(r"\d{6}[ -]\d{7}"),                        # 주민등록번호 형태
)


@dataclass(frozen=True)
class CleanConfig:
    min_dwell: float = 20.0        # 이 미만의 클릭은 만족이 아니라 바운스
    transfer_reformulations: bool = True  # 실패한 앞 쿼리를 세션의 최종 만족 문서에 연결
    skip_above: int = 2            # 만족 클릭 위에서 스킵된 문서를 hard negative로 (최대 n개)
    min_count: int = 1             # (쿼리, 문서) 집계 최소 등장 횟수 — 대량 로그에선 2+ 권장


@dataclass
class CleanResult:
    # {"query", "doc_id", "count", "negatives": [doc_id, ...]} — doc_id는 corpus 해석 전
    pairs: list[dict] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _satisfied_clicks(event: dict, min_dwell: float) -> list[dict]:
    return [
        c for c in (event.get("clicks") or [])
        if c.get("doc_id") and float(c.get("dwell_sec") or 0) >= min_dwell
    ]


def clean(events: list[dict], config: CleanConfig = CleanConfig()) -> CleanResult:
    """이벤트 → 신뢰할 수 있는 (쿼리, 문서) 쌍 + 규칙별 리포트.

    규칙 순서: PII 드롭 → 만족 클릭 판별(dwell) → 세션 정리(마지막 만족 클릭이
    세션의 결론; 실패한 앞 쿼리는 그 결론으로 전이) → skip-above hard negative →
    (쿼리, 문서) 집계.
    """
    report = {
        "events": len(events),
        "dropped_pii": 0,
        "bounces_ignored": 0,
        "positives_direct": 0,
        "positives_transferred": 0,
        "hard_negatives": 0,
        "abandoned_sessions": 0,
    }

    # 세션별로 모은다 — 재검색(전이) 판단은 세션 단위여야 한다.
    sessions: dict[str, list[dict]] = {}
    for i, event in enumerate(events):
        query = str(event.get("query") or "").strip()
        if not query:
            continue
        if contains_pii(query):
            report["dropped_pii"] += 1
            continue
        sessions.setdefault(str(event.get("session") or f"__solo-{i}"), []).append(event)

    raw_pairs: list[dict] = []   # {"query", "doc_id", "negatives"}
    for session_events in sessions.values():
        resolved: list[tuple[dict, dict]] = []   # (event, 만족 클릭)
        unresolved: list[dict] = []              # 만족 클릭이 없는 이벤트
        for event in session_events:
            satisfied = _satisfied_clicks(event, config.min_dwell)
            report["bounces_ignored"] += len(event.get("clicks") or []) - len(satisfied)
            if satisfied:
                # 같은 이벤트에 만족 클릭이 여럿이면 마지막 것이 결론 (앞은 경유지)
                resolved.append((event, satisfied[-1]))
            else:
                unresolved.append(event)

        if not resolved:
            report["abandoned_sessions"] += 1
            continue

        for event, click in resolved:
            negatives: list[str] = []
            rank = click.get("rank")
            results = event.get("results") or []
            if config.skip_above and isinstance(rank, int) and rank > 1:
                # 만족 클릭 위의 문서는 사용자가 보고 지나친 것 — 그 쿼리의 hard negative
                negatives = [d for d in results[: rank - 1] if d != click["doc_id"]]
                negatives = negatives[-config.skip_above:]
                report["hard_negatives"] += len(negatives)
            raw_pairs.append({
                "query": event["query"].strip(),
                "doc_id": click["doc_id"],
                "negatives": negatives,
            })
            report["positives_direct"] += 1

        if config.transfer_reformulations:
            # 세션의 결론 = 마지막 만족 클릭 문서. 만족 클릭 없이 끝난(=엔진이 못 푼)
            # 쿼리를 그 문서에 연결한다 — 재검색으로만 도달 가능한 표현의 supervision.
            final_doc = resolved[-1][1]["doc_id"]
            resolved_queries = {e["query"].strip().lower() for e, _ in resolved}
            for event in unresolved:
                query = event["query"].strip()
                if query.lower() in resolved_queries:
                    continue  # 같은 쿼리가 이미 직접 쌍을 얻었다면 전이 불필요
                raw_pairs.append({"query": query, "doc_id": final_doc, "negatives": []})
                report["positives_transferred"] += 1

    # (쿼리, 문서) 집계 — 등장 횟수는 신뢰도, negatives는 합집합.
    grouped: dict[tuple[str, str], dict] = {}
    for pair in raw_pairs:
        key = (pair["query"].lower(), pair["doc_id"])
        entry = grouped.setdefault(
            key, {"query": pair["query"], "doc_id": pair["doc_id"], "count": 0, "negatives": []}
        )
        entry["count"] += 1
        for doc_id in pair["negatives"]:
            if doc_id not in entry["negatives"] and doc_id != pair["doc_id"]:
                entry["negatives"].append(doc_id)

    pairs = [p for p in grouped.values() if p["count"] >= config.min_count]
    report["unique_pairs"] = len(pairs)
    report["below_min_count"] = len(grouped) - len(pairs)
    return CleanResult(pairs=pairs, report=report)


def naive_pairs(events: list[dict]) -> list[dict]:
    """현행 인입과 같은 순진한 변환 — 모든 클릭이 쌍이 된다 (리허설의 대조군)."""
    out = []
    for event in events:
        query = str(event.get("query") or "").strip()
        if not query:
            continue
        for click in event.get("clicks") or []:
            if click.get("doc_id"):
                out.append({"query": query, "doc_id": click["doc_id"], "negatives": []})
    return out
