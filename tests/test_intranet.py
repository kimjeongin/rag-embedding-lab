"""가상 인트라넷 데이터셋 생성기 — 양성 대조군의 전제 조건을 검증한다."""
from __future__ import annotations

import pytest

from rag.datagen.intranet import PAGE_KINDS, POLICIES, SYSTEMS, generate


@pytest.fixture(scope="module")
def dataset():
    return generate(seed=20260718)


def test_catalog_covers_every_system_page_and_policy(dataset):
    pages, _, _ = dataset
    assert len(pages) == len(SYSTEMS) * len(PAGE_KINDS) + len(POLICIES)
    urls = [p["url"] for p in pages]
    assert len(urls) == len(set(urls))                    # 페이지 단위 url, 중복 없음
    for page in pages:
        assert page["description"]
        assert page["agent_prompt"]
        assert set(page["metadata"]) == {"collected_by", "version", "collected_at", "source"}


def test_aliases_never_appear_in_the_corpus(dataset):
    """대조군의 전제: 은어→시스템 연결은 학습쌍에만 있다 — corpus에 새면 실험 무효."""
    pages, train, eval_pairs = dataset
    aliases = [a for sys in SYSTEMS for a in sys.aliases]
    corpus_text = " ".join(
        " ".join((p["title"], p["description"], p["content"])) for p in pages
    )
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
