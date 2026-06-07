"""`rag-train` — fine-tune the embedding model (rag.training.train)."""
from __future__ import annotations

from rag.training.config import TrainingConfig
from rag.training.train import train


def main() -> None:
    train(TrainingConfig.from_env())
