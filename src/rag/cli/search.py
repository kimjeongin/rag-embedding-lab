"""`rag-search` — one query against the live Qdrant index (serving smoke test).

    uv run rag-search "vpn 안됨"
    uv run rag-search "연차 신청" --model outputs/embedding-ft -k 5

The query MUST be embedded by the same model that built the index — the dim guard in
rag.serving catches size mismatches, but a same-dim different model would silently
rank garbage, so pass --model (or ST_MODEL) deliberately.
"""
from __future__ import annotations

import argparse
import asyncio

from rag import lab, serving
from rag.config import Settings
from rag.embeddings import build_embedder
from rag.vectorstore.qdrant import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="search the live serving index")
    parser.add_argument("query")
    parser.add_argument("--model", default=Settings.from_env().st_model,
                        help="sentence-transformers model dir (default: ST_MODEL env)")
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("--truncate-dim", type=int, default=None)
    args = parser.parse_args()

    dim = lab.infer_dim("sentence-transformers", args.model, "", args.truncate_dim)
    settings = lab.build_eval_settings("sentence-transformers", args.model, dim, "", args.truncate_dim)

    async def run() -> dict:
        with QdrantStore(settings.qdrant_url) as store:
            async with build_embedder(settings) as embedder:
                return await serving.search(settings, embedder, store, args.query, args.top_k)

    result = asyncio.run(run())
    print(f"[search] \"{result['query']}\" · {result['collection']} · {result['model']}")
    for rank, hit in enumerate(result["hits"], 1):
        title = hit.get("title") or "(제목 없음)"
        print(f"  {rank:>2}. {hit['score']:.4f}  {title}")
        if hit.get("url"):
            print(f"       {hit['url']}")
