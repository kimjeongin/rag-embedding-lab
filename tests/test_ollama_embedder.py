"""OllamaEmbedder request batching — a MockTransport stands in for the server.

The regression this guards: an eval corpus (hundreds of long docs) sent as ONE
/api/embed request exceeds the client timeout. The adapter must slice the inputs
so the timeout bounds a single slice, while callers still see one flat result.
"""
import json

import httpx

from rag.config import Settings
from rag.core.entities import Document
from rag.embeddings.ollama import _BATCH, OllamaEmbedder


def _fake_server(seen_batches: list[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        seen_batches.append(len(inputs))
        # embedding = [index-within-request, 0.0] — enough to check order/dim plumbing
        return httpx.Response(200, json={"embeddings": [[float(i), 0.0] for i in range(len(inputs))]})

    return handler


async def test_embed_slices_large_inputs_and_concatenates():
    seen: list[int] = []
    settings = Settings(embed_dim=2, embedder="ollama")
    async with httpx.AsyncClient(transport=httpx.MockTransport(_fake_server(seen))) as http:
        embedder = OllamaEmbedder(http, settings)
        docs = [Document(content=f"doc {i}") for i in range(_BATCH * 2 + 3)]
        rows = await embedder.embed_documents(docs)

    assert seen == [_BATCH, _BATCH, 3]          # sliced, not one giant request
    assert len(rows) == _BATCH * 2 + 3          # …but the caller sees one flat list
    assert rows[_BATCH] == [0.0, 0.0]           # slice boundaries line up (no overlap/loss)


async def test_embed_small_input_is_a_single_request():
    seen: list[int] = []
    settings = Settings(embed_dim=2, embedder="ollama")
    async with httpx.AsyncClient(transport=httpx.MockTransport(_fake_server(seen))) as http:
        embedder = OllamaEmbedder(http, settings)
        await embedder.embed_queries(["연차 규정"])
    assert seen == [1]
