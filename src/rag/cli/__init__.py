"""Command-line entrypoints — every runnable command lives here (one per file).

These are thin: they read config from the environment and delegate to the real
logic in the other packages. Wired to console scripts in pyproject.toml.

  rag-serve          serve.py        — run the HTTP API (rag.api.app)
  rag-gen-data       gen_data.py     — toy dataset            (rag.datagen.dummy)
  rag-gen-synthetic  gen_synthetic.py— LLM-generated dataset  (rag.datagen.synthetic)
  rag-train          train.py        — fine-tune              (rag.training.train)
  rag-eval           evaluate.py     — measure retrieval      (rag.evaluation.retrieval)
"""
