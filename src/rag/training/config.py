"""Training configuration (separate from the serving ``rag.config``).

Same pattern as ``rag.config.Settings``: a frozen dataclass with module-constant
defaults and a ``from_env()`` factory, so env is read explicitly at call time (not
at import). The dataset paths come from ``rag.dataset`` (the shared dataset contract),
not defined here, so datagen/evaluation don't have to reach into the training config.
``query_instruction`` defaults to the serving value (train/inference parity).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from rag.core.formatting import DEFAULT_QUERY_INSTRUCTION
from rag.dataset import DEFAULT_EVAL_FILE, DEFAULT_TRAIN_FILE, dataset_paths

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_OUTPUT_DIR = "outputs/embedding-ft"
_DEFAULT_EPOCHS = 1
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_LR = 2e-5


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    base_model: str = _DEFAULT_BASE_MODEL       # served model's HF checkpoint
    output_dir: str = _DEFAULT_OUTPUT_DIR
    train_file: str = DEFAULT_TRAIN_FILE         # dataset location (rag.dataset)
    eval_file: str = DEFAULT_EVAL_FILE
    epochs: int = _DEFAULT_EPOCHS
    batch_size: int = _DEFAULT_BATCH_SIZE        # bigger → more in-batch negatives
    learning_rate: float = _DEFAULT_LR
    device: str = ""                             # "" = auto (cuda → mps → cpu)
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Build config from environment variables, falling back to the defaults."""
        train_file, eval_file = dataset_paths()
        return cls(
            base_model=os.getenv("TRAIN_BASE_MODEL", _DEFAULT_BASE_MODEL),
            output_dir=os.getenv("TRAIN_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR),
            train_file=train_file,
            eval_file=eval_file,
            epochs=int(os.getenv("TRAIN_EPOCHS", str(_DEFAULT_EPOCHS))),
            batch_size=int(os.getenv("TRAIN_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE))),
            learning_rate=float(os.getenv("TRAIN_LR", str(_DEFAULT_LR))),
            device=os.getenv("TRAIN_DEVICE", ""),
            query_instruction=os.getenv("QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
        )
