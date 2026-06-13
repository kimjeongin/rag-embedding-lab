"""Build the BEIR eval set FROM the real corpus + the held-out test pairs.

The sample eval (``eval_corpus``) fakes a haystack with synthetic distractors. Once
a real corpus exists (``rag-crawl``), the honest haystack is simply THE WHOLE SITE:
every crawled page goes into the eval corpus, the held-out test split's queries
become the eval queries, and each query's source page is its gold document. The
train-split pages sit in the corpus as natural distractors — exactly the production
situation, where the index always contains every page.

``build`` is the pure transform; ``load_and_build`` adds the file IO + validation so
the CLI (``rag-gen-eval`` with ``EVAL_SOURCE=corpus``) and the API route share ONE
implementation and only translate its ``ValueError`` into their own error surface
(SystemExit / HTTP 400). The dev/final qrels split stays in ``eval_corpus.split_qrels``.
"""
from __future__ import annotations


def build(
    corpus_docs: list[dict], test_pairs: list[dict]
) -> tuple[list[dict], list[dict], list[tuple[str, str, int]], int]:
    """(corpus, queries, qrels_rows, skipped) in BEIR shapes.

    Pages get stable ids by corpus order (``page-N``). Each test pair resolves back
    to its source page by exact (title, content) — synthetic positives are verbatim
    copies of corpus docs, so equality is the join key. Pairs that don't resolve
    (the corpus was re-crawled after the pairs were generated) are skipped and
    counted; a non-zero count means "regenerate the pairs".
    """
    corpus = [
        {"_id": f"page-{i}", "title": doc.get("title"), "text": doc["content"]}
        for i, doc in enumerate(corpus_docs)
    ]
    page_ids = {
        (doc.get("title"), doc["content"]): f"page-{i}" for i, doc in enumerate(corpus_docs)
    }

    queries: list[dict] = []
    qrels: list[tuple[str, str, int]] = []
    skipped = 0
    for j, pair in enumerate(test_pairs):
        positive = pair.get("positive") or {}
        doc_id = page_ids.get((positive.get("title"), positive.get("content", "")))
        if doc_id is None:
            skipped += 1
            continue
        query_id = f"q-test-{j}"
        queries.append({"_id": query_id, "text": pair["query"]})
        qrels.append((query_id, doc_id, 1))
    return corpus, queries, qrels, skipped


def load_and_build(
    corpus_file: str, test_file: str
) -> tuple[list[dict], list[dict], list[tuple[str, str, int]], int]:
    """``build`` over the files on disk, with the validation both entrypoints need.

    Raises ``ValueError`` (Korean, user-facing) when the inputs are missing/empty or
    nothing matches — the caller maps it to its layer's error type.
    """
    from rag.dataset import load_jsonl

    try:
        docs = list(load_jsonl(corpus_file))
        test_pairs = list(load_jsonl(test_file))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{exc.filename}이 없습니다 — rag-crawl과 rag-gen-synthetic을 먼저 실행하세요"
        ) from exc
    if not docs or not test_pairs:
        raise ValueError("corpus 또는 test 데이터가 비어 있습니다 — 크롤과 학습쌍 생성을 먼저 실행하세요")

    corpus, queries, qrels, skipped = build(docs, test_pairs)
    if not qrels:
        raise ValueError("test 쌍이 corpus와 매칭되지 않습니다 — corpus를 재크롤했다면 학습쌍부터 재생성하세요")
    return corpus, queries, qrels, skipped
