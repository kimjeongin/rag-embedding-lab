"""Fine-tune the embedding model (the logic behind ``rag-train``).

Contrastive fine-tuning with sentence-transformers:
  - loss: selectable (mnrl | cached_mnrl | gist | triplet) — all four fit the
    (query, positive[, negatives]) dataset; triplet requires the negatives.
  - eval: InformationRetrievalEvaluator + eval_loss on the held-out test split,
    every epoch.
  - early stopping: ``epochs`` is a ceiling. Each epoch the monitored metric
    (val nDCG@10 or val loss) is checked; the best epoch's weights are snapshotted
    and training stops after ``early_stop_patience`` epochs without improvement.
    What gets saved is the BEST epoch, and the final directory name says so
    (…-mnrl-e7). patience=0 restores the old run-everything-save-last behaviour.

Runs on macOS (MPS), Ubuntu (CUDA), or CPU — device is auto-detected
(rag.training.model.pick_device). Heavy imports are inside ``train`` so the module
stays importable without the training stack.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

from rag.modelprofile import resolve_profile
from rag.training.config import TrainingConfig
from rag.training.data import to_ir_eval, to_training_dataset
from rag.training.model import load_base_model, pick_device

LOSSES = ("mnrl", "cached_mnrl", "gist", "triplet")


def _num(value: float | None) -> str:
    """Format a metric for the parseable [epoch] line — '-' when missing."""
    return "-" if value is None else f"{value:.4f}"


def _compose_output_dir(cfg: TrainingConfig, saved_epoch: int) -> str:
    """The final save path: ``{output_dir}-{loss}[-r{rank}]-e{saved_epoch}``.

    Named at the END of training because only then is the best epoch known. A path
    that already exists gets ``-2``, ``-3``, … so reruns never overwrite a finished
    model. auto_name=False returns output_dir verbatim (the old behaviour).
    """
    if not cfg.auto_name:
        return cfg.output_dir
    lora = f"-r{cfg.lora_r}" if cfg.method == "lora" else ""
    mrl = "-mrl" if cfg.matryoshka else ""
    base = f"{cfg.output_dir}-{cfg.loss}{mrl}{lora}-e{saved_epoch}"
    path, n = base, 2
    while Path(path).exists():
        path, n = f"{base}-{n}", n + 1
    return path


def resolve_matryoshka_dims(requested: tuple[int, ...], model_dim: int) -> list[int]:
    """The truncation dimensions to train, descending. A request is honoured but
    clamped to dims the model can actually produce (≤ model_dim); an empty/unusable
    request falls back to halving from the full dim (d, d/2, d/4, …) down to 64."""
    if requested:
        usable = sorted({d for d in requested if 0 < d <= model_dim}, reverse=True)
        if usable:
            return usable
    auto, d = [], model_dim
    while d >= 64:
        auto.append(d)
        d //= 2
    return auto


def _build_loss(model, cfg: TrainingConfig, has_negatives: bool):
    """The training loss, chosen by ``cfg.loss`` — then optionally wrapped in
    MatryoshkaLoss. All four base losses read the same dataset columns ((anchor,
    positive) or (anchor, positive, negative))."""
    from sentence_transformers import SentenceTransformer, losses

    if cfg.loss == "mnrl":
        # InfoNCE with in-batch negatives — the de-facto standard for retrieval.
        base = losses.MultipleNegativesRankingLoss(model)
    elif cfg.loss == "cached_mnrl":
        # Same objective as MNRL, but GradCache lets batch_size grow far beyond
        # device memory — and for MNRL, batch size IS the number of negatives.
        base = losses.CachedMultipleNegativesRankingLoss(model)
    elif cfg.loss == "gist":
        # MNRL where a guide model vetoes in-batch "negatives" that are actually
        # relevant to the query (false negatives corrupt the contrastive signal).
        print(f"[train] GIST guide model: {cfg.gist_guide}")
        guide = SentenceTransformer(cfg.gist_guide, device=pick_device(cfg.device))
        base = losses.GISTEmbedLoss(model, guide)
    elif cfg.loss == "triplet":
        if not has_negatives:
            raise SystemExit(
                "[train] TripletLoss에는 hard negative가 필요합니다 — "
                "데이터 탭에서 hard-negative mining을 켜고 데이터를 다시 생성하세요"
            )
        # Embeddings are L2-normalized, so cosine distance with a small margin
        # (the euclidean default margin=5 would never be satisfied on a unit sphere).
        base = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.05,
        )
    else:
        raise SystemExit(f"[train] unknown TRAIN_LOSS={cfg.loss!r} — expected one of {LOSSES}")

    if cfg.matryoshka:
        # Train the loss at several embedding prefixes at once, so a truncated vector
        # is still well-ordered. Wraps any base loss — the contrastive objective is
        # unchanged, just applied to [:d] for each d.
        dims = resolve_matryoshka_dims(cfg.matryoshka_dims, model.get_embedding_dimension())
        print(f"[train] Matryoshka representation learning — dims {dims}")
        return losses.MatryoshkaLoss(model, base, matryoshka_dims=dims)
    return base


def _best_epoch_callback(model, cfg: TrainingConfig):
    """The practitioner loop, written out: validate every epoch, snapshot the best
    weights, stop when `patience` epochs pass without improvement.

    Hooks into the trainer via ``on_evaluate`` (fires after each epoch's eval, with
    eval_loss and the IR evaluator's metrics merged into one dict). The snapshot is
    a plain ``state_dict`` → ``best.pt``, identical for full and LoRA runs (a LoRA
    state_dict simply contains the adapter tensors). patience=0 = observe only.
    """
    import torch
    from transformers import TrainerCallback

    class BestEpochCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_epoch = 0
            self.best_value: float | None = None
            self.bad_epochs = 0
            self.history: list[dict] = []
            self.snapshot = Path(cfg.output_dir) / "best.pt"

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            eval_loss = metrics.get("eval_loss")
            ndcg = next((v for k, v in metrics.items() if k.endswith("ndcg@10")), None)
            epoch = int(round(state.epoch or len(self.history) + 1))
            self.history.append({"epoch": epoch, "eval_loss": eval_loss, "ndcg": ndcg})

            value = ndcg if cfg.early_stop_metric == "ndcg" else eval_loss
            improved = value is not None and (
                self.best_value is None
                or (value > self.best_value if cfg.early_stop_metric == "ndcg" else value < self.best_value)
            )
            if improved:
                self.best_epoch, self.best_value, self.bad_epochs = epoch, value, 0
                if cfg.early_stop_patience > 0:
                    self.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), self.snapshot)
            else:
                self.bad_epochs += 1

            print(
                f"[epoch] {epoch}/{cfg.epochs} eval_loss={_num(eval_loss)} "
                f"ndcg@10={_num(ndcg)} best={self.best_epoch}"
            )
            if cfg.early_stop_patience > 0 and self.bad_epochs >= cfg.early_stop_patience:
                control.should_training_stop = True
                print(
                    f"[train] early stopping — {self.bad_epochs} epochs without "
                    f"improvement (patience={cfg.early_stop_patience})"
                )
            return control

    return BestEpochCallback()


def train(cfg: TrainingConfig) -> dict:
    """Run fine-tuning; return the post-training eval metrics."""
    import torch
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    device = pick_device(cfg.device)
    print(f"[train] method={cfg.method}  loss={cfg.loss}  device={device}  base_model={cfg.base_model}")

    model = load_base_model(cfg)
    if cfg.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        # "attention" uses Qwen/LLaMA-style projection names (our default base model);
        # "all-linear" is the architecture-agnostic fallback that always matches.
        target_modules = (
            ["q_proj", "k_proj", "v_proj", "o_proj"] if cfg.lora_target == "attention" else "all-linear"
        )
        # `Transformer.auto_model` is a read-only property over `.model`, so wrap `.model`.
        transformer = model[0]
        transformer.model = get_peft_model(
            transformer.model,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=target_modules,
            ),
        )
        model.to(device)  # land the freshly-added adapter params on the train device
        print(
            f"[train] LoRA enabled — r={cfg.lora_r} alpha={cfg.lora_alpha} "
            f"dropout={cfg.lora_dropout} target={cfg.lora_target}"
        )
        transformer.model.print_trainable_parameters()
    # TripletLoss takes exactly (anchor, positive, negative); the MNRL/GIST family
    # treats EVERY extra column as another negative — so give it all mined ones.
    max_negatives = 1 if cfg.loss == "triplet" else cfg.max_negatives
    # The base model's input format — every text below (train, validation) is built
    # with it, and it's recorded in train_meta.json so serving inherits the same one.
    profile = resolve_profile(cfg.base_model, cfg.model_profile)
    print(f"[train] input format profile: {profile.name}"
          f"{'' if profile.uses_instruction else ' (instruction unused by this model)'}")
    train_dataset = to_training_dataset(
        cfg.train_file, cfg.query_instruction, max_negatives, profile=profile
    )
    negative_columns = [c for c in train_dataset.column_names if c.startswith("negative")]
    print(f"[train] {len(train_dataset)} training pairs from {cfg.train_file} "
          f"(hard negatives per pair: {len(negative_columns)})")

    loss = _build_loss(model, cfg, has_negatives=bool(negative_columns))

    # Validation ranks the held-out queries against held-out docs PLUS the train-split
    # docs as distractors — a val corpus of only the few test docs saturates nDCG@10
    # near 1.0 and early stopping would be steering on noise. (Costs one corpus embed
    # per epoch; at lab scale that's seconds.)
    queries, corpus, relevant = to_ir_eval(
        cfg.eval_file, cfg.query_instruction, distractor_file=cfg.train_file, profile=profile
    )
    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant, name="val", show_progress_bar=False
    )
    print(f"[train] validation: {len(queries)} queries over {len(corpus)} docs "
          f"(train-split docs included as distractors)")
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
        per_device_eval_batch_size=cfg.batch_size,  # same batches every epoch → comparable eval_loss
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,           # fixed → a config is reproducible; sweep seeds to see variance
        warmup_ratio=0.1,        # ST-recommended LR ramp-up (the HF default is none)
        logging_steps=logging_steps,
        # MNRL treats every other in-batch positive as a negative. Our data has many
        # queries per document, so the default sampler routinely puts a query's own
        # positive in the batch again — as a "negative" — corrupting the signal.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Re-run the evaluator every epoch: a falling loss with a falling nDCG is
        # overfitting, invisible to the single before/after measurement.
        eval_strategy="epoch",
        save_strategy="no",      # best-epoch snapshotting is ours (best.pt), not HF's
        report_to=[],            # no wandb/tensorboard
        # Recompute activations in the backward pass instead of storing them. This is
        # the ONLY memory lever that leaves the recipe intact — batch size changes the
        # in-batch negatives MNRL trains on, and LoRA changes which weights move, so
        # both make a run non-comparable to a full-precision baseline. Costs ~30% time.
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    # eval_dataset feeds eval_strategy="epoch" (HF requires one even with an
    # evaluator) and adds a held-out eval_loss next to the evaluator's nDCG. Same
    # negative arity as training, so eval_loss is computed on the same task shape.
    eval_dataset = to_training_dataset(
        cfg.eval_file, cfg.query_instruction, max_negatives, profile=profile
    )
    best_cb = _best_epoch_callback(model, cfg)
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        loss=loss, evaluator=evaluator, callbacks=[best_cb],
    )
    trainer.train()

    ran = int(round(trainer.state.epoch or 0)) or cfg.epochs
    early_stopped = ran < cfg.epochs
    saved_epoch = ran
    if cfg.early_stop_patience > 0 and best_cb.best_epoch and best_cb.snapshot.exists():
        if best_cb.best_epoch != ran:
            model.load_state_dict(torch.load(best_cb.snapshot, map_location="cpu"))
            print(f"[train] restored best epoch {best_cb.best_epoch} weights (last was epoch {ran})")
        saved_epoch = best_cb.best_epoch
        best_cb.snapshot.unlink()
        with contextlib.suppress(OSError):
            best_cb.snapshot.parent.rmdir()  # drop the staging dir if nothing else is in it

    if cfg.method == "lora":
        # Merge the adapter into the base weights so the saved model is a plain
        # SentenceTransformer — serving (ES, etc.) never needs to know LoRA was used.
        model[0].model = model[0].model.merge_and_unload()
        print("[train] merged LoRA adapter into base weights (standalone model)")

    final_dir = _compose_output_dir(cfg, saved_epoch)
    model.save_pretrained(final_dir)
    _write_meta(final_dir, cfg, best_cb.history, saved_epoch, ran, early_stopped)
    print(f"[train] summary best_epoch={saved_epoch} ran={ran} early_stopped={'yes' if early_stopped else 'no'}")
    print(f"[train] saved fine-tuned model to {final_dir}")

    metrics = evaluator(model)
    print("[train] eval (after fine-tuning):")
    _print_metrics(metrics)
    return metrics


def _write_meta(final_dir: str, cfg: TrainingConfig, history: list[dict],
                saved_epoch: int, ran: int, early_stopped: bool) -> None:
    """train_meta.json — the full recipe behind the model name's short suffix."""
    from rag.dataset import file_fingerprint

    meta: dict = {
        "base_model": cfg.base_model,
        # The input format this model was trained with. Serving reads it back
        # (rag.modelprofile) so a tuned model can't be served with another model's
        # prompts — the failure that produces no error, only worse numbers.
        "model_profile": resolve_profile(cfg.base_model, cfg.model_profile).name,
        "method": cfg.method,
        "loss": cfg.loss,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dropout": cfg.dropout,
        "seed": cfg.seed,
        "max_epochs": cfg.epochs,
        "epochs_ran": ran,
        "saved_epoch": saved_epoch,
        "early_stopped": early_stopped,
        "monitor": cfg.early_stop_metric,
        "patience": cfg.early_stop_patience,
        # Content hashes of the data this model was trained/validated on — the files
        # get regenerated in place, so this is the only durable answer to "what data
        # trained this model?" (eval sets make the same move with their fingerprint).
        "train_data_fingerprint": file_fingerprint(cfg.train_file),
        "val_data_fingerprint": file_fingerprint(cfg.eval_file),
        "history": history,                  # per-epoch {epoch, eval_loss, ndcg}
    }
    if cfg.note.strip():
        meta["note"] = cfg.note.strip()
    if cfg.method == "lora":
        meta |= {
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target": cfg.lora_target,
        }
    if cfg.loss == "gist":
        meta["gist_guide"] = cfg.gist_guide
    if cfg.matryoshka:
        # The exact dims are derived at train time from the model's own embedding dim;
        # record the request — the resolved list is in the training log either way.
        meta["matryoshka"] = True
        meta["matryoshka_dims"] = list(cfg.matryoshka_dims)
    Path(final_dir, "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _print_metrics(metrics: dict) -> None:
    for key in sorted(metrics):
        if any(k in key for k in ("accuracy@1", "ndcg@10", "mrr@10")):
            print(f"           {key} = {metrics[key]:.4f}")
