"""클릭로그 노이즈 시뮬레이터 — 실로그 도착 전, 클리닝 규칙의 리허설 상대.

인트라넷 우주(가상 회사 '다온') 위에 **정답을 아는** 세션 클릭로그를 생성한다.
각 세션은 하나의 의도(시스템 × 요구)를 갖고, 현실 로그의 노이즈를 확률 혼합으로
재현한다:

  - 만족 클릭 (정답, 긴 dwell) — 다수 트래픽
  - 포지션 바이어스 (오답 1등을 충분히 긴 dwell로 클릭 — 겉보기 만족, 진짜 노이즈)
  - 오클릭 바운스 (짧은 dwell 후 정답 클릭)
  - 무클릭 이탈
  - **재검색 세션**: 은어 쿼리는 현행(어휘 일치) 엔진이 못 풀어 노출에 정답이
    없고, 사용자가 공식명으로 고쳐 검색해 성공한다 — 은어 supervision은 클릭이
    아니라 이 세션 구조에만 존재한다 (클리닝의 전이 규칙이 회수해야 할 몫).
  - PII 쿼리 (이메일·전화번호 섞임) — 드롭돼야 할 몫.

`simulate`는 (events, truth)를 돌려준다. truth는 쿼리 텍스트 → (정답 doc_id,
슬라이스)로, 클리너가 뽑은 쌍의 정밀도/재현율을 채점하는 데 쓴다 — 실로그에는
없는 사치지만, 그래서 리허설이다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from rag.datagen.intranet import SYSTEMS, generate

# 로그 쿼리 템플릿 — intranet.py의 평가 템플릿과 겹치지 않게 유지한다 (로그에서
# 뽑은 쌍으로 학습해 평가하는 실험이 표현 암기로 오염되지 않도록).
STANDARD_LOG_TEMPLATES = (
    "{name} {task} 어디",
    "{task} 하고 싶은데 {name}",
    "{name} {task} 화면 안 보임",
    "{name}에서 {task} 메뉴",
)
JARGON_LOG_TEMPLATES = (
    "{alias} {task} 안돼",
    "{alias} 어디 갔지",
    "{alias} {task} 하는데 막힘",
    "{alias} 링크 좀",
)
REFORMULATED_TEMPLATES = (
    "{name} {task}",
    "{name} {task} 방법",
)


@dataclass(frozen=True)
class SimConfig:
    n_sessions: int = 600
    p_jargon_session: float = 0.35   # 은어로 시작하는(재검색이 필요한) 세션 비율
    p_position_bias: float = 0.12    # 오답 1등을 겉보기-만족 dwell로 클릭
    p_misclick: float = 0.15         # 짧은 dwell 오클릭 후 정답 클릭
    p_abandon: float = 0.08          # 무클릭 이탈
    p_pii: float = 0.03              # 쿼리에 PII가 섞이는 비율


def _click(doc_id: str, rank: int, dwell: float) -> dict:
    return {"doc_id": doc_id, "rank": rank, "dwell_sec": round(dwell, 1)}


def _id_index() -> dict[tuple[str, str], str]:
    """eval corpus와 같은 규칙(page-N = 생성 순서)으로 (system, kind) → doc_id."""
    pages, _, _ = generate()
    return {(p["system"], p["kind"]): f"page-{i}" for i, p in enumerate(pages)}


def _serp(gold: str | None, sys_slug: str, kind: str,
          id_by_key: dict[tuple[str, str], str], rng: random.Random,
          gold_rank: int | None) -> list[str]:
    """노출 10건 — 같은 kind의 타 시스템 페이지(동형 distractor) 위주로 채운다."""
    same_kind = [d for (slug, k), d in id_by_key.items() if k == kind and slug != sys_slug]
    same_system = [d for (slug, k), d in id_by_key.items() if slug == sys_slug and k != kind]
    pool = rng.sample(same_kind, min(8, len(same_kind))) + rng.sample(
        same_system, min(3, len(same_system))
    )
    results = [d for d in pool if d != gold][:10]
    if gold is not None and gold_rank is not None:
        results.insert(min(gold_rank - 1, len(results)), gold)
    return results[:10]


def simulate(
    seed: int = 20260720, config: SimConfig = SimConfig()
) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """(events, truth) — truth: 쿼리 텍스트(lower) → (정답 doc_id, slice)."""
    rng = random.Random(seed)
    id_by_key = _id_index()
    events: list[dict] = []
    truth: dict[str, tuple[str, str]] = {}

    def emit(session: str, query: str, results: list[str], clicks: list[dict]) -> dict:
        event = {"session": session, "query": query, "results": results, "clicks": clicks}
        events.append(event)
        return event

    for s in range(config.n_sessions):
        session = f"s-{s:05d}"
        sys = rng.choice(SYSTEMS)
        task = rng.choice(sys.tasks)
        kind = rng.choice(("guide", "guide", "guide", "access", "faq"))
        gold = id_by_key[(sys.slug, kind)]
        task_word = task.name if kind == "guide" else rng.choice(("권한", "계정", "오류"))

        if rng.random() < config.p_jargon_session:
            # ── 재검색 세션: 은어 쿼리(엔진 실패, 노출에 정답 없음) → 공식명 재검색 ──
            jargon_query = rng.choice(JARGON_LOG_TEMPLATES).format(
                alias=rng.choice(sys.aliases), task=task_word
            )
            truth[jargon_query.lower()] = (gold, "jargon")
            first = emit(session, jargon_query,
                         _serp(None, sys.slug, kind, id_by_key, rng, None), [])
            if rng.random() < 0.4:  # 오답 1등을 잠깐 눌러봤다 바로 나옴
                first["clicks"] = [_click(first["results"][0], 1, rng.uniform(2, 8))]
            if rng.random() < 0.15:
                continue  # 재검색조차 안 하고 이탈 — 이 은어는 로그로 회수 불가
            query = rng.choice(REFORMULATED_TEMPLATES).format(name=sys.name, task=task.name)
            truth[query.lower()] = (gold, "standard")
            gold_rank = rng.choice((1, 1, 2, 3))
            emit(session, query, _serp(gold, sys.slug, kind, id_by_key, rng, gold_rank),
                 [_click(gold, gold_rank, rng.uniform(40, 280))])
            continue

        # ── 표준 세션 ──
        query = rng.choice(STANDARD_LOG_TEMPLATES).format(name=sys.name, task=task_word)
        if rng.random() < config.p_pii:
            # PII 쿼리는 truth에 넣지 않는다 — 회수 대상이 아니라 차단 대상
            pii_query = f"{query} {rng.choice(('kim.cs@daon.example', '010-4821-7733'))}"
            emit(session, pii_query, _serp(gold, sys.slug, kind, id_by_key, rng, 1),
                 [_click(gold, 1, rng.uniform(40, 200))])
            continue

        truth[query.lower()] = (gold, "standard")
        roll = rng.random()
        if roll < config.p_abandon:
            gold_rank = rng.choice((1, 2, 3))
            emit(session, query, _serp(gold, sys.slug, kind, id_by_key, rng, gold_rank), [])
        elif roll < config.p_abandon + config.p_position_bias:
            # 오답 1등을 겉보기-만족 dwell로 클릭 — 클리너가 못 거르는 '진짜' 노이즈.
            gold_rank = rng.choice((2, 3, 4))  # 정답을 밀어 1등이 항상 오답이 되게
            event = emit(session, query,
                         _serp(gold, sys.slug, kind, id_by_key, rng, gold_rank), [])
            event["clicks"] = [_click(event["results"][0], 1, rng.uniform(25, 90))]
        elif roll < config.p_abandon + config.p_position_bias + config.p_misclick:
            gold_rank = rng.choice((1, 1, 2, 3, 4))
            event = emit(session, query,
                         _serp(gold, sys.slug, kind, id_by_key, rng, gold_rank), [])
            wrong = rng.choice([d for d in event["results"] if d != gold])
            event["clicks"] = [
                _click(wrong, event["results"].index(wrong) + 1, rng.uniform(2, 9)),
                _click(gold, gold_rank, rng.uniform(40, 280)),
            ]
        else:
            gold_rank = rng.choice((1, 1, 1, 2, 2, 3, 4))
            emit(session, query, _serp(gold, sys.slug, kind, id_by_key, rng, gold_rank),
                 [_click(gold, gold_rank, rng.uniform(30, 300))])

    return events, truth
