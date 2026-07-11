"""`rag-index` — embed the crawled corpus into Qdrant and repoint the live alias.

The serving side of the lab: run it after training (or after a new crawl) to make the
fine-tuned model searchable. Serving is sentence-transformers in-process by decision —
no Ollama on this path — so the embedder is always the ST backend and the embedding
dim is read off the model (no manual EMBED_DIM to get wrong).

    uv run rag-index --model outputs/embedding-ft            # index data/corpus.jsonl
    uv run rag-index --model outputs/... --recreate           # force a rebuild
    uv run rag-index --model outputs/... --prune              # drop old rollback copies

Same model + same corpus content = same collection name, so re-running is a no-op:
safe to automate (e.g. right after a handoff/crawl). Env: QDRANT_URL / QDRANT_COLLECTION.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from rag import lab, serving
from rag.config import Settings
from rag.embeddings import build_embedder
from rag.vectorstore.qdrant import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="corpus.jsonl → Qdrant serving index")
    parser.add_argument("--model", default=Settings.from_env().st_model,
                        help="sentence-transformers model dir (default: ST_MODEL env)")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--batch", type=int, default=32, help="docs per embed+upsert batch")
    parser.add_argument("--truncate-dim", type=int, default=None,
                        help="Matryoshka: index truncated vectors (model must be trained for it)")
    parser.add_argument("--recreate", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--prune", action="store_true",
                        help="after the swap, delete the family's other (rollback) collections")
    args = parser.parse_args()

    dim = lab.infer_dim("sentence-transformers", args.model, "", args.truncate_dim)
    settings = lab.build_eval_settings("sentence-transformers", args.model, dim, "", args.truncate_dim)

    async def run() -> dict:
        with QdrantStore(settings.qdrant_url) as store:
            async with build_embedder(settings) as embedder:
                started = time.perf_counter()

                def progress(done: int, total: int) -> None:
                    rate = done / max(time.perf_counter() - started, 1e-9)
                    print(f"  {done}/{total} docs ({rate:.1f}/s)", flush=True)

                summary = await serving.index_corpus(
                    settings, embedder, store, args.corpus,
                    recreate=args.recreate, batch_size=args.batch, progress=progress,
                )
            if args.prune:
                for name in serving.prune_collections(settings, store):
                    print(f"  pruned {name}")
            return summary

    summary = asyncio.run(run())
    verb = "already up to date" if summary["skipped"] else "indexed"
    print(f"[index] {verb}: {summary['docs']} docs · dim {summary['dim']} · model {summary['model']}")
    print(f"  {summary['alias']} → {summary['collection']}")
