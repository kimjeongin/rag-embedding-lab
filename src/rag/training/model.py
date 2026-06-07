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


def load_base_model(cfg: TrainingConfig):
    """Load the base model to fine-tune as a SentenceTransformer on the picked device."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.base_model, device=pick_device(cfg.device))
