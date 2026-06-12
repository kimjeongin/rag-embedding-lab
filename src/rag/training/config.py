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
_DEFAULT_EPOCHS = 12                             # a *ceiling* — early stopping ends sooner
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_LR = 2e-5
_DEFAULT_LOSS = "mnrl"                           # mnrl | cached_mnrl | gist | triplet
_DEFAULT_GIST_GUIDE = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_PATIENCE = 3                            # epochs without improvement before stopping
_DEFAULT_MONITOR = "ndcg"                        # "ndcg" (val nDCG@10) | "loss" (val loss)
_DEFAULT_METHOD = "full"                         # "full" (all params) | "lora" (adapters)
_DEFAULT_LORA_R = 16
_DEFAULT_LORA_ALPHA = 32
_DEFAULT_LORA_DROPOUT = 0.05
_DEFAULT_LORA_TARGET = "all-linear"              # "all-linear" | "attention" (q/k/v/o only)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    base_model: str = _DEFAULT_BASE_MODEL       # served model's HF checkpoint
    output_dir: str = _DEFAULT_OUTPUT_DIR
    train_file: str = DEFAULT_TRAIN_FILE         # dataset location (rag.dataset)
    eval_file: str = DEFAULT_EVAL_FILE
    epochs: int = _DEFAULT_EPOCHS                # max epochs (early stopping may end sooner)
    batch_size: int = _DEFAULT_BATCH_SIZE        # bigger → more in-batch negatives
    learning_rate: float = _DEFAULT_LR
    device: str = ""                             # "" = auto (cuda → mps → cpu)
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    # Loss function — all four work with the (query, positive[, negatives]) dataset;
    # triplet additionally *requires* hard negatives on every record.
    loss: str = _DEFAULT_LOSS
    gist_guide: str = _DEFAULT_GIST_GUIDE        # guide model for GISTEmbedLoss

    # Backbone dropout override. None = keep the model's defaults. The config key
    # differs per architecture (BERT: hidden_dropout_prob, Qwen: attention_dropout, …);
    # rag.training.model detects whichever keys exist and sets them all.
    dropout: float | None = None

    # Early stopping: run up to `epochs`, validate every epoch, stop after `patience`
    # epochs without improvement on the monitored metric, and save the BEST epoch's
    # weights (not the last). patience=0 disables both — every epoch runs and the
    # final weights are saved, as before.
    early_stop_patience: int = _DEFAULT_PATIENCE
    early_stop_metric: str = _DEFAULT_MONITOR    # "ndcg" (higher=better) | "loss" (lower=better)

    # Append "-{loss}[-r{r}]-e{best_epoch}" to output_dir at save time, so the model
    # name itself says how it was trained (the full config goes to train_meta.json).
    auto_name: bool = True

    # Small-data fine-tunes vary run to run; a fixed seed makes a config reproducible,
    # and sweeping the SAME config over several seeds measures that variance.
    seed: int = 42

    # Experimenter's hypothesis/memo ("lr을 올리면?") — recorded in train_meta.json
    # and shown next to the run in Compare, so labels don't have to carry prose.
    note: str = ""

    # Fine-tuning method. "full" updates every weight; "lora" trains small low-rank
    # adapters (faster, less memory) that are merged back into the base on save, so the
    # output is a normal model either way (serving doesn't need to know which was used).
    method: str = _DEFAULT_METHOD
    lora_r: int = _DEFAULT_LORA_R                # adapter rank (lora-only)
    lora_alpha: int = _DEFAULT_LORA_ALPHA        # adapter scaling (lora-only)
    lora_dropout: float = _DEFAULT_LORA_DROPOUT  # adapter dropout (lora-only)
    lora_target: str = _DEFAULT_LORA_TARGET      # which layers get adapters (lora-only)

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Build config from environment variables, falling back to the defaults."""
        train_file, eval_file = dataset_paths()
        dropout = os.getenv("TRAIN_DROPOUT", "")
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
            loss=os.getenv("TRAIN_LOSS", _DEFAULT_LOSS),
            gist_guide=os.getenv("TRAIN_GIST_GUIDE", _DEFAULT_GIST_GUIDE),
            dropout=float(dropout) if dropout else None,   # "" = keep model defaults
            early_stop_patience=int(os.getenv("TRAIN_PATIENCE", str(_DEFAULT_PATIENCE))),
            early_stop_metric=os.getenv("TRAIN_MONITOR", _DEFAULT_MONITOR),
            auto_name=os.getenv("TRAIN_AUTO_NAME", "1").lower() not in ("0", "false", "no"),
            seed=int(os.getenv("TRAIN_SEED", "42")),
            note=os.getenv("TRAIN_NOTE", ""),
            method=os.getenv("TRAIN_METHOD", _DEFAULT_METHOD),
            lora_r=int(os.getenv("TRAIN_LORA_R", str(_DEFAULT_LORA_R))),
            lora_alpha=int(os.getenv("TRAIN_LORA_ALPHA", str(_DEFAULT_LORA_ALPHA))),
            lora_dropout=float(os.getenv("TRAIN_LORA_DROPOUT", str(_DEFAULT_LORA_DROPOUT))),
            lora_target=os.getenv("TRAIN_LORA_TARGET", _DEFAULT_LORA_TARGET),
        )
