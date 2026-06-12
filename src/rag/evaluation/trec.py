"""TREC run format — import an EXTERNAL retriever's ranking as a registry run.

The standard interchange format for rankings (trec_eval):

    query-id  Q0  doc-id  rank  score  tag

Why the lab wants it: production already runs BM25 + hybrid + rerank. The lab stays
dense-only on purpose (variable isolation), but the question "does this dense model
add anything BM25 doesn't already find?" needs BM25's ranking — so the serving team
exports one TREC run file for the eval queries, the lab scores it against the same
qrels, and from then on it's an ordinary registry row: the paired diff view shows
complementarity (queries dense wins that BM25 misses, and vice versa) for free.
"""
from __future__ import annotations


def parse_trec_run(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """({query_id: [doc_id, …] best-first}, errors) from TREC run lines.

    Whitespace-separated, 6 columns; ordered by the rank column (the score column is
    ignored — ranks are authoritative). Malformed lines are reported, not fatal.
    """
    ranked: dict[str, list[tuple[int, str]]] = {}
    errors: list[str] = []
    for lineno, line in enumerate(text.strip().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 4:
            errors.append(f"{lineno}행: 열이 부족합니다 (query-id Q0 doc-id rank score tag)")
            continue
        query_id, doc_id, raw_rank = parts[0], parts[2], parts[3]
        try:
            rank = int(float(raw_rank))
        except ValueError:
            errors.append(f"{lineno}행: rank가 숫자가 아닙니다 ({raw_rank!r})")
            continue
        ranked.setdefault(query_id, []).append((rank, doc_id))

    rankings: dict[str, list[str]] = {}
    for query_id, pairs in ranked.items():
        pairs.sort(key=lambda p: p[0])
        seen: set[str] = set()
        ordered: list[str] = []
        for _, doc_id in pairs:
            if doc_id not in seen:
                seen.add(doc_id)
                ordered.append(doc_id)
        rankings[query_id] = ordered
    return rankings, errors
