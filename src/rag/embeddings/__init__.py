"""Embedding *inference* adapters + the factory that builds the configured one.

  - ``OllamaEmbedder``              — the original model via Ollama (default).
  - ``SentenceTransformerEmbedder`` — a local (e.g. fine-tuned) model.
  - ``build_embedder(settings)``    — async CM yielding the one ``Settings.embedder`` picks.

Both adapters implement rag.core.ports.Embedder. Callers use ``build_embedder``;
the concrete classes are constructed only inside the factory.
"""
from rag.embeddings.factory import build_embedder
from rag.embeddings.ollama import OllamaEmbedder
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = ["build_embedder", "OllamaEmbedder", "SentenceTransformerEmbedder"]
