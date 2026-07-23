"""가상 인트라넷 데이터셋 생성기 — 양성 대조군의 전제 조건을 검증한다."""
from __future__ import annotations

import pytest

from rag.datagen.intranet import PAGE_KINDS, POLICIES, SYSTEMS, generate


@pytest.fixture(scope="module")
def dataset():
    return generate(seed=20260718)


# 실 운영 컬렉션 payload 스키마 — corpus 레코드가 이 필드를 모두 실어야 한다.
PAYLOAD_FIELDS = (
    "site_id", "url", "version_name", "title", "title_eng", "llm_title",
    "description", "description_eng", "user_queries", "need_steps", "hard_guide_name",
)


def test_catalog_covers_every_system_page_and_policy(dataset):
    pages, _, _ = dataset
    assert len(pages) == len(SYSTEMS) * len(PAGE_KINDS) + len(POLICIES)
    urls = [p["url"] for p in pages]
    assert len(urls) == len(set(urls))                    # 페이지 단위 url, 중복 없음
    for page in pages:
        assert page["description"]
        assert page["agent_prompt"]
        assert set(page["metadata"]) == {"collected_by", "version", "collected_at", "source"}


def test_records_carry_the_production_payload_schema(dataset):
    """리허설 corpus는 실 운영 검색 컬렉션의 payload 스키마를 그대로 실어야 한다."""
    pages, _, _ = dataset
    for page in pages:
        for field in PAYLOAD_FIELDS:
            assert field in page, f"{page['url']}에 payload 필드 '{field}' 누락"
        assert isinstance(page["user_queries"], list) and page["user_queries"]
        assert isinstance(page["need_steps"], bool)
        assert page["title_eng"] and page["description_eng"] and page["llm_title"]
        # 절차형 페이지는 권위 가이드명을 가리키고, 아니면 None
        assert (page["hard_guide_name"] is not None) == page["need_steps"]
    # site_id는 사이트(시스템) 단위로 공유되고 페이지 단위 url은 그 아래 여럿
    assert len({p["site_id"] for p in pages}) == len(SYSTEMS) + len(POLICIES)


def test_aliases_never_appear_in_the_corpus(dataset):
    """대조군의 전제: 은어→시스템 연결은 학습쌍에만 있다 — corpus에 새면 실험 무효.

    다국어·LLM 확장 필드(title_eng·llm_title·user_queries 등)까지 포함해 검사한다.
    """
    pages, train, eval_pairs = dataset
    aliases = [a for sys in SYSTEMS for a in sys.aliases]
    text_fields = ("title", "title_eng", "llm_title", "description",
                   "description_eng", "content", "hard_guide_name", "version_name")
    parts: list[str] = []
    for p in pages:
        parts.extend(str(p.get(f) or "") for f in text_fields)
        parts.extend(p["user_queries"])
    corpus_text = " ".join(parts)
    for alias in aliases:
        assert alias not in corpus_text
    jargon_train = [r for r in train if r["slice"] == "jargon"]
    assert any(any(a in r["query"] for a in aliases) for r in jargon_train)


def test_train_and_eval_queries_do_not_overlap(dataset):
    _, train, eval_pairs = dataset
    assert not ({r["query"] for r in train} & {r["query"] for r in eval_pairs})


def test_eval_pairs_carry_both_slices_and_resolve_to_pages(dataset):
    pages, _, eval_pairs = dataset
    slices = {r["slice"] for r in eval_pairs}
    assert slices == {"standard", "jargon"}
    keys = {(p["title"], p["content"]) for p in pages}
    for pair in eval_pairs:
        assert (pair["positive"]["title"], pair["positive"]["content"]) in keys


def test_generation_is_deterministic():
    assert generate(seed=7) == generate(seed=7)
