"""From-scratch BM25 — the lab's lexical baseline, for measuring dense의 한계 기여.

프로덕션 검색은 BM25 + dense + 리랭커 하이브리드다. dense 모델의 가치는 "혼자 잘
하는가"가 아니라 **BM25가 놓치는 쿼리에서 후보를 구제하는가**(한계 기여)로 재야
하이브리드의 실제 개선과 방향이 맞는다 — 사내 시스템명·약어는 문자 일치라 BM25가
이미 잘 잡는 영역이고, 그 위에서 dense recall만 보면 기여가 중복 계산된다.

구현 노트:
  - Okapi BM25 (k1=1.2, b=0.75), 순수 stdlib — ``rag.evaluation.metrics``와 같은
    입장: 외부 의존 없이 단위 테스트 가능한 참조 구현.
  - 토큰화는 **문자 bigram**: 형태소 분석기 없이 한국어 어절의 조사·어미를 이기는
    표준 트릭이다 ("머니핀에서"와 "머니핀"이 bigram 수준에서 겹친다). 한 글자
    토큰은 그대로 쓴다. 프로덕션 BM25(형태소 기반)와 절대값은 다르지만, 어휘
    일치가 잡는 것/못 잡는 것의 구조는 같아 상보성 측정용 대역으로 충분하다.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[0-9a-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word chunks → character bigrams (single-char chunks kept whole)."""
    out: list[str] = []
    for tok in _WORD.findall(text.lower()):
        if len(tok) == 1:
            out.append(tok)
        else:
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return out


class BM25:
    """BM25 index over {doc_id: text}; ``search`` returns best-first doc ids."""

    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter[str]] = {d: Counter(tokenize(t)) for d, t in docs.items()}
        self.doc_len = {d: sum(c.values()) for d, c in self.tf.items()}
        n = len(docs)
        self.avg_len = (sum(self.doc_len.values()) / n) if n else 0.0
        df: Counter[str] = Counter()
        for counts in self.tf.values():
            df.update(counts.keys())
        # BM25+-style smoothed idf (always positive, trec_eval-compatible shape)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def score(self, query: str, doc_id: str) -> float:
        counts = self.tf[doc_id]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1.0))
        total = 0.0
        for term in tokenize(query):
            f = counts.get(term)
            if f:
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query: str, top_k: int = 10) -> list[str]:
        scored = [(self.score(query, d), d) for d in self.tf]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))  # 점수 동률은 id로 안정 정렬
        return [d for s, d in scored[:top_k] if s > 0]


def rank_eval_corpus(
    corpus: dict[str, dict[str, str | None]],
    queries: dict[str, str],
    top_k: int = 10,
) -> dict[str, list[str]]:
    """{query_id: best-first doc ids} — BEIR 로더 산출물을 그대로 받아 랭킹한다.

    문서 텍스트는 dense 쪽과 같은 재료(title + 본문)를 쓴다 — 두 랭커가 같은
    입력을 봐야 상보성 비교가 공정하다.
    """
    index = BM25({
        doc_id: f"{doc.get('title') or ''} {doc['text']}" for doc_id, doc in corpus.items()
    })
    return {qid: index.search(text, top_k) for qid, text in queries.items()}
