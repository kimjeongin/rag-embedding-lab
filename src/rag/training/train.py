"""Fine-tune the embedding model (the logic behind ``rag-train``).

Contrastive fine-tuning with sentence-transformers:
  - loss: MultipleNegativesRankingLoss (InfoNCE with in-batch negatives) — needs
    only (query, positive) pairs.
  - eval: InformationRetrievalEvaluator on the held-out test split.

Runs on macOS (MPS), Ubuntu (CUDA), or CPU — device is auto-detected
(rag.training.model.pick_device). Heavy imports are inside ``train`` so the module
stays importable without the training stack.
"""
from __future__ import annotations

from rag.training.config import TrainingConfig
from rag.training.data import to_ir_eval, to_training_dataset
from rag.training.model import load_base_model, pick_device


def train(cfg: TrainingConfig) -> dict:
    """Run fine-tuning; return the post-training eval metrics."""
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    device = pick_device(cfg.device)
    print(f"[train] device={device}  base_model={cfg.base_model}")

    model = load_base_model(cfg)
    train_dataset = to_training_dataset(cfg.train_file, cfg.query_instruction)
    print(f"[train] {len(train_dataset)} training pairs from {cfg.train_file}")

    loss = MultipleNegativesRankingLoss(model)

    queries, corpus, relevant = to_ir_eval(cfg.eval_file, cfg.query_instruction)
    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant, name="dummy-test", show_progress_bar=False
    )
    print("[train] baseline eval (before fine-tuning):")
    _print_metrics(evaluator(model))

    # Log often enough that the loss curve has points even on short runs: aim for ~50
    # points regardless of dataset size (a 3-step toy run logs every step; a 5k-step run
    # every ~100). A fixed logging_steps=10 left short runs with an empty curve.
    steps_per_epoch = max(1, -(-len(train_dataset) // cfg.batch_size))  # ceil division
    total_steps = steps_per_epoch * cfg.epochs
    logging_steps = max(1, total_steps // 50)

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=logging_steps,
        save_strategy="no",      # save once at the end, no intermediate checkpoints
        report_to=[],            # no wandb/tensorboard
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss, evaluator=evaluator
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    print(f"[train] saved fine-tuned model to {cfg.output_dir}")

    metrics = evaluator(model)
    print("[train] eval (after fine-tuning):")
    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: dict) -> None:
    for key in sorted(metrics):
        if any(k in key for k in ("accuracy@1", "ndcg@10", "mrr@10")):
            print(f"           {key} = {metrics[key]:.4f}")
