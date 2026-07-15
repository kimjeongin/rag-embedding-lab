"""Lab support — default model selection + truncate-dim plumbing (the rest of rag.lab
needs Ollama/torch and isn't unit-tested)."""
import pytest

from rag.lab import build_eval_settings, default_model, infer_dim


def test_default_model_prefers_an_embedding_model_for_ollama():
    assert default_model("ollama", ["qwen3:4b", "qwen3-embedding:0.6b"]) == "qwen3-embedding:0.6b"


def test_infer_dim_short_circuits_to_truncate_dim():
    # truncate_dim is the produced dimension — no model load, no Ollama call
    assert infer_dim("sentence-transformers", "outputs/whatever", "", truncate_dim=256) == 256
    assert infer_dim("ollama", "m", "http://x", truncate_dim=128) == 128


def test_build_eval_settings_carries_truncate_dim_for_st():
    s = build_eval_settings("sentence-transformers", "outputs/m", 256, "", truncate_dim=256)
    assert s.truncate_dim == 256 and s.embed_dim == 256 and s.st_model == "outputs/m"


def test_build_eval_settings_rejects_truncate_for_ollama():
    # Ollama embeds at a fixed dimension — truncation is a sentence-transformers feature
    with pytest.raises(ValueError, match="truncate_dim"):
        build_eval_settings("ollama", "m", 256, "http://x", truncate_dim=256)


def test_default_model_falls_back_to_first_for_ollama_without_embedding():
    assert default_model("ollama", ["a", "b"]) == "a"


def test_default_model_first_dir_for_sentence_transformers():
    assert default_model("sentence-transformers", ["outputs/ft", "x"]) == "outputs/ft"


def test_default_model_prefers_the_serving_model_when_available():
    choices = ["outputs/a", "outputs/b"]
    assert default_model("sentence-transformers", choices, preferred="outputs/b") == "outputs/b"
    # the serving model may be gone from outputs/ (deleted) — fall back gracefully
    assert default_model("sentence-transformers", choices, preferred="outputs/gone") == "outputs/a"


def test_default_model_empty_when_no_choices():
    assert default_model("ollama", []) == ""
