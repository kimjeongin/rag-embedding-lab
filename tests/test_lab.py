"""Lab support — default model selection (the rest of rag.lab needs Ollama/torch and
isn't unit-tested)."""
from rag.lab import default_model


def test_default_model_prefers_an_embedding_model_for_ollama():
    assert default_model("ollama", ["qwen3:4b", "qwen3-embedding:0.6b"]) == "qwen3-embedding:0.6b"


def test_default_model_falls_back_to_first_for_ollama_without_embedding():
    assert default_model("ollama", ["a", "b"]) == "a"


def test_default_model_first_dir_for_sentence_transformers():
    assert default_model("sentence-transformers", ["outputs/ft", "x"]) == "outputs/ft"


def test_default_model_empty_when_no_choices():
    assert default_model("ollama", []) == ""
