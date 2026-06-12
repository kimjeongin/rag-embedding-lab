"""Base model loading + cross-platform device selection.

ML libraries are imported lazily *inside* functions so importing this module (and
``rag.training``) never requires the training stack to be installed.
"""
from __future__ import annotations

from rag.training.config import TrainingConfig


def pick_device(preferred: str = "") -> str:
    """Choose a device that works on both macOS and Ubuntu.

    Order: explicit override → CUDA (Ubuntu GPU) → MPS (Apple Silicon) → CPU.
    """
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Dropout lives under a different config key per architecture. We set every key the
# model's config actually has — the printout says which, so a run is never ambiguous
# about what "dropout=0.1" meant for this particular backbone.
_DROPOUT_KEYS = (
    "hidden_dropout_prob",           # BERT family
    "attention_probs_dropout_prob",  # BERT family
    "attention_dropout",             # Qwen / LLaMA family
    "hidden_dropout",                # e.g. Falcon
    "resid_pdrop",                   # GPT-2 family
    "embd_pdrop",                    # GPT-2 family
)


def load_base_model(cfg: TrainingConfig):
    """Load the base model to fine-tune as a SentenceTransformer on the picked device."""
    from sentence_transformers import SentenceTransformer

    config_kwargs = {}
    if cfg.dropout is not None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(cfg.base_model)
        applied = [k for k in _DROPOUT_KEYS if hasattr(config, k)]
        # config_kwargs (not model_kwargs): these are AutoConfig overrides — passed as
        # model kwargs they'd reach the model constructor and TypeError.
        config_kwargs = dict.fromkeys(applied, cfg.dropout)
        print(f"[train] dropout={cfg.dropout} → {', '.join(applied) or '(no dropout keys on this architecture)'}")

    return SentenceTransformer(
        cfg.base_model, device=pick_device(cfg.device), config_kwargs=config_kwargs or None
    )
