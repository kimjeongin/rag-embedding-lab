"""_point_to_hit mapping — pure, no Qdrant server needed."""
from rag.stores.qdrant import _point_to_hit


class FakePoint:
    """Stand-in for a qdrant_client ScoredPoint."""

    def __init__(self, id_, score, payload):
        self.id = id_
        self.score = score
        self.payload = payload


def test_maps_score_as_similarity_and_surfaces_metadata():
    point = FakePoint(
        3,
        0.87,
        {
            "content": "body text",
            "metadata": {
                "url": "https://e.com/p",
                "domain": "e.com",
                "path": "/p",
                "title": "T",
            },
        },
    )
    hit = _point_to_hit(point)
    assert hit.id == 3
    assert hit.content == "body text"
    assert hit.similarity == 0.87  # cosine score IS the similarity
    assert hit.url == "https://e.com/p"
    assert hit.domain == "e.com"
    assert hit.title == "T"


def test_handles_missing_payload():
    hit = _point_to_hit(FakePoint(1, 0.5, None))
    assert hit.id == 1
    assert hit.content == ""
    assert hit.domain is None
