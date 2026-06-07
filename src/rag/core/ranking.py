"""Pure ranking/grouping rules over Hit entities (no I/O, trivially unit-testable).

These are the post-processing building blocks for the two search modes. They take
and return domain entities, so they can be tested without a database or a model.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from rag.core.entities import Hit, Site


def apply_max_per_domain(hits: Sequence[Hit], max_per_domain: int | None) -> list[Hit]:
    """Keep at most `max_per_domain` hits per domain, preserving the incoming
    similarity order. `None` means no cap. Hits sharing a domain (incl. a missing
    domain, keyed as None) count against the same bucket."""
    if max_per_domain is None:
        return list(hits)
    counts: dict[str | None, int] = {}
    kept: list[Hit] = []
    for hit in hits:
        if counts.get(hit.domain, 0) < max_per_domain:
            kept.append(hit)
            counts[hit.domain] = counts.get(hit.domain, 0) + 1
    return kept


def site_score(similarities: Sequence[float]) -> float:
    """Score a site from its member pages' similarities.

    Currently the best (max) page. Isolated so it can later become e.g. a top-N
    average without touching the grouping/sorting code.
    """
    return max(similarities)


def group_by_site(hits: Sequence[Hit], top_k: int) -> list[Site]:
    """Group hits by domain into ranked Sites.

    `hits` arrive in similarity-desc order, so each domain bucket (an OrderedDict
    preserves first-seen order) is already page-sorted. Sites are scored via
    `site_score`, sorted by that score desc, and truncated to `top_k`.
    """
    buckets: "OrderedDict[str | None, list[Hit]]" = OrderedDict()
    for hit in hits:
        buckets.setdefault(hit.domain, []).append(hit)

    sites = [
        Site(
            domain=domain,
            score=site_score([h.similarity for h in pages]),
            pages=sorted(pages, key=lambda h: h.similarity, reverse=True),
        )
        for domain, pages in buckets.items()
    ]
    sites.sort(key=lambda s: s.score, reverse=True)
    return sites[:top_k]
