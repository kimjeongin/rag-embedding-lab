"""Use cases exercised with in-memory fakes — no Ollama, no vector DB.

That these run at all is the point of the refactor: IndexDocuments / SearchDocuments
depend on the Embedder / VectorStore *ports*, so fakes implementing those Protocols
are enough to test the business logic.
"""
from collections.abc import Sequence

from rag.core.entities import Document, EmbeddedDocument, Hit
from rag.usecases import IndexDocuments, SearchDocuments


class FakeEmbedder:
    """Structurally implements rag.core.ports.Embedder."""

    def __init__(self) -> None:
        self.seen_documents: list[Document] | None = None

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        self.seen_documents = list(documents)
        return [[float(i)] for i, _ in enumerate(documents)]

    async def embed_query(self, query: str) -> list[float]:
        return [0.0]


class FakeStore:
    """Structurally implements rag.core.ports.VectorStore."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.hits = hits or []
        self.added: list[EmbeddedDocument] | None = None
        self.search_limit: int | None = None

    async def add(self, documents: Sequence[EmbeddedDocument]) -> list[int]:
        self.added = list(documents)
        return list(range(1, len(self.added) + 1))

    async def search(self, embedding: Sequence[float], limit: int) -> list[Hit]:
        self.search_limit = limit
        return self.hits[:limit]

    async def count(self) -> int:
        return len(self.hits)


def hit(id_: int, domain: str, sim: float) -> Hit:
    return Hit(id=id_, content=f"c{id_}", similarity=sim, domain=domain,
               url=f"https://{domain}/{id_}", title=f"t{id_}")


async def test_index_derives_metadata_and_stores_raw_body():
    embedder, store = FakeEmbedder(), FakeStore()
    ids = await IndexDocuments(embedder, store).execute(
        [Document(content="body", url="https://e.com/p", title="T")]
    )
    assert ids == [1]
    assert embedder.seen_documents[0].title == "T"          # title available to embedder
    stored = store.added[0]
    assert isinstance(stored, EmbeddedDocument)
    assert stored.content == "body"                          # raw body, no title
    assert stored.metadata["domain"] == "e.com"             # derived server-side
    assert stored.metadata["path"] == "/p"


async def test_search_pages_no_cap_limits_to_top_k():
    store = FakeStore([hit(1, "a", 0.9), hit(2, "a", 0.8), hit(3, "b", 0.7)])
    res = await SearchDocuments(FakeEmbedder(), store).pages("q", top_k=2)
    assert [h.id for h in res] == [1, 2]
    assert store.search_limit == 2                           # no cap -> limit == top_k


async def test_search_pages_with_cap_pulls_fetch_k_then_diversifies():
    store = FakeStore([hit(1, "a", 0.9), hit(2, "a", 0.8), hit(3, "b", 0.7), hit(4, "c", 0.6)])
    res = await SearchDocuments(FakeEmbedder(), store).pages(
        "q", top_k=3, max_per_domain=1, fetch_k=10
    )
    assert store.search_limit == 10                          # cap -> pulls fetch_k pool
    assert [h.id for h in res] == [1, 3, 4]                   # one per domain, up to top_k


async def test_search_sites_groups_and_ranks():
    hits = sorted([hit(1, "a", 0.9), hit(2, "b", 0.8), hit(3, "a", 0.95)],
                  key=lambda h: h.similarity, reverse=True)
    store = FakeStore(hits)
    sites = await SearchDocuments(FakeEmbedder(), store).sites("q", top_k=2, fetch_k=50)
    assert store.search_limit == 50
    assert [s.domain for s in sites] == ["a", "b"]
    assert sites[0].score == 0.95
