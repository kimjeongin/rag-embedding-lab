"""Settings — embedder selection from env (pure, no torch/Ollama)."""
from rag.config import Settings


def test_defaults_use_sentence_transformers():
    s = Settings()
    assert s.embedder == "sentence-transformers"
    assert s.st_model == "outputs/embedding-ft"
    assert s.embed_dim == 1024


def test_from_env_selects_sentence_transformers(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "sentence-transformers")
    monkeypatch.setenv("ST_MODEL", "outputs/my-ft")
    monkeypatch.setenv("EMBED_DIM", "384")
    s = Settings.from_env()
    assert s.embedder == "sentence-transformers"
    assert s.st_model == "outputs/my-ft"
    assert s.embed_dim == 384


def test_active_model_follows_embedder():
    assert Settings(embedder="ollama").active_model == "qwen3-embedding:0.6b"
    assert Settings(embedder="sentence-transformers", st_model="outputs/ft").active_model == "outputs/ft"
