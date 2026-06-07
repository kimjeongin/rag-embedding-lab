"""Pure ranking rules — no DB, no model."""
from rag.core.entities import Hit
from rag.core.ranking import apply_max_per_domain, group_by_site, site_score


def hit(id_: int, domain: str, sim: float) -> Hit:
    return Hit(id=id_, content=f"c{id_}", similarity=sim, domain=domain,
               url=f"https://{domain}/{id_}", title=f"t{id_}")


def test_max_per_domain_none_keeps_all():
    hits = [hit(1, "a", 0.9), hit(2, "a", 0.8), hit(3, "b", 0.7)]
    assert apply_max_per_domain(hits, None) == hits


def test_max_per_domain_caps_and_preserves_order():
    hits = [hit(1, "a", 0.9), hit(2, "a", 0.8), hit(3, "b", 0.7), hit(4, "a", 0.6)]
    assert [h.id for h in apply_max_per_domain(hits, 1)] == [1, 3]


def test_site_score_is_max():
    assert site_score([0.2, 0.9, 0.5]) == 0.9


def test_group_by_site_orders_sites_and_pages_within():
    # arrives similarity-desc (as the store returns it)
    hits = sorted([hit(1, "a", 0.9), hit(2, "b", 0.8), hit(3, "a", 0.95), hit(4, "b", 0.1)],
                  key=lambda h: h.similarity, reverse=True)
    sites = group_by_site(hits, top_k=2)
    assert [s.domain for s in sites] == ["a", "b"]   # a's best 0.95 > b's best 0.8
    assert sites[0].score == 0.95
    assert [p.id for p in sites[0].pages] == [3, 1]   # pages sorted desc within site


def test_group_by_site_truncates_to_top_k():
    hits = [hit(1, "a", 0.9), hit(2, "b", 0.8), hit(3, "c", 0.7)]
    assert [s.domain for s in group_by_site(hits, top_k=2)] == ["a", "b"]
