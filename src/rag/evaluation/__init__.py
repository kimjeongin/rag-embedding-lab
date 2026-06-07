"""Offline retrieval evaluation — measure how well an embedder retrieves.

Loads a BEIR-format eval set (corpus + queries + qrels — see `beir`), ranks the corpus
for each query with the configured embedder (`retrieval`), and reports
recall@k / MRR@10 / nDCG@10 (`metrics`) so you can compare models (e.g. base vs a
fine-tuned one). The CLI entrypoint is ``rag-eval``; the data contract and the
experiment's assumptions are documented in docs/evaluation.md.
"""
