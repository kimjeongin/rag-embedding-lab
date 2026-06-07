"""Configuration as an injectable value, not a global.

`Settings` is a frozen dataclass built ONCE at the composition root
(`rag.api.app.create_app`) via `Settings.from_env()` and passed to the adapters
that need it. Nothing else reads `os.environ`, so dependencies are explicit and
tests can construct a `Settings(...)` directly instead of patching globals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from rag.core.formatting import DEFAULT_QUERY_INSTRUCTION

# Defaults live as module constants so they're a single source shared by both the
# field defaults and `from_env` (and not read off the slotted class, where the
# class attribute is a descriptor rather than the value).
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_QDRANT_COLLECTION = "documents"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
_DEFAULT_EMBED_DIM = 1024
_DEFAULT_EMBEDDER = "ollama"
_DEFAULT_ST_MODEL = "outputs/embedding-ft"


@dataclass(frozen=True, slots=True)
class Settings:
    qdrant_url: str = _DEFAULT_QDRANT_URL
    qdrant_collection: str = _DEFAULT_QDRANT_COLLECTION
    embed_dim: int = _DEFAULT_EMBED_DIM
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    # Which embedding backend to use:
    #   "ollama"                -> OllamaEmbedder (default; the original model)
    #   "sentence-transformers" -> SentenceTransformerEmbedder (e.g. a fine-tuned model)
    embedder: str = _DEFAULT_EMBEDDER

    # Ollama backend
    ollama_url: str = _DEFAULT_OLLAMA_URL
    embed_model: str = _DEFAULT_EMBED_MODEL

    # sentence-transformers backend (model path/name + device; "" device = auto)
    st_model: str = _DEFAULT_ST_MODEL
    st_device: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to the defaults."""
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", _DEFAULT_QDRANT_COLLECTION),
            embed_dim=int(os.getenv("EMBED_DIM", str(_DEFAULT_EMBED_DIM))),
            query_instruction=os.getenv("QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
            embedder=os.getenv("EMBEDDER", _DEFAULT_EMBEDDER),
            ollama_url=os.getenv("OLLAMA_URL", _DEFAULT_OLLAMA_URL),
            embed_model=os.getenv("EMBED_MODEL", _DEFAULT_EMBED_MODEL),
            st_model=os.getenv("ST_MODEL", _DEFAULT_ST_MODEL),
            st_device=os.getenv("ST_DEVICE", ""),
        )

    @property
    def active_model(self) -> str:
        """The embedding model actually in use (Ollama name or ST model path)."""
        return self.st_model if self.embedder == "sentence-transformers" else self.embed_model
