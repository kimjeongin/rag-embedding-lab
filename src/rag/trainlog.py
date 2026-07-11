"""Parse a training run's stdout into structured progress — framework-free.

``rag-train`` (a subprocess) streams the HF Trainer's logs: per-step ``{'loss': …,
'epoch': …}`` dicts, a per-epoch ``[epoch] n/max eval_loss=… ndcg@10=… best=…``
validation line, a baseline/after ``ndcg@10 = …`` line on either side of the
fine-tune, and a final ``[train] summary``/``saved fine-tuned model to`` pair. The
HTTP API's SSE stream turns that text into points and numbers for the live Train
screen, so the regex lives here once — stdlib ``re`` only, framework-free and
unit-testable.

Tolerant of transformers' quoted (``'loss': '0.39'``) and unquoted (``'loss': 0.39``)
forms, and of tqdm's ``\\r`` progress redraws.
"""
from __future__ import annotations

import re

_LOSS_RE = re.compile(r"'loss':\s*'?([0-9.eE+-]+)")
_EPOCH_RE = re.compile(r"'epoch':\s*'?([0-9.eE+-]+)")
_NDCG_RE = re.compile(r"ndcg@10\s*=\s*([0-9.]+)")
# [epoch] 3/12 eval_loss=0.4123 ndcg@10=0.9312 best=3   ('-' = metric missing)
_EPOCH_LINE_RE = re.compile(
    r"\[epoch\]\s+(\d+)/(\d+)\s+eval_loss=([0-9.eE+-]+)\s+ndcg@10=([0-9.eE+-]+)\s+best=(\d+)"
)
_SAVED_RE = re.compile(r"saved fine-tuned model to (.+?)\s*$")
_SUMMARY_RE = re.compile(r"\[train\] summary best_epoch=(\d+) ran=(\d+) early_stopped=(yes|no)")


def clean_tqdm(text: str) -> str:
    """Collapse tqdm carriage-return redraws so each line reads as its final state."""
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def parse_loss_line(line: str) -> dict | None:
    """One line's training-step log as ``{epoch, loss}`` (epoch None when the line
    doesn't carry one), or None for a non-loss line. The streaming job runner parses
    each line as it arrives — re-parsing the whole accumulated log per line would be
    quadratic over a long run."""
    loss = _LOSS_RE.search(line)
    if not loss:
        return None
    epoch = _EPOCH_RE.search(line)
    return {
        "epoch": float(epoch.group(1)) if epoch else None,
        "loss": float(loss.group(1)),
    }


def loss_point(parsed: dict, step: int) -> dict:
    """A ``parse_loss_line`` result as the stored ``{step, epoch, loss}`` point
    (a missing epoch falls back to the step number, keeping the curve monotonic)."""
    return {
        "step": step,
        "epoch": parsed["epoch"] if parsed["epoch"] is not None else float(step),
        "loss": parsed["loss"],
    }


def parse_loss_points(text: str) -> list[dict]:
    """Every logged training step as ``{step, epoch, loss}`` (step is 1-based order)."""
    points: list[dict] = []
    for line in text.split("\n"):
        parsed = parse_loss_line(line)
        if parsed:
            points.append(loss_point(parsed, step=len(points) + 1))
    return points


def _opt_float(raw: str) -> float | None:
    """A metric printed as '-' means it was missing that epoch."""
    try:
        return float(raw)
    except ValueError:
        return None


def parse_epoch_line(line: str) -> dict | None:
    """One per-epoch validation line as ``{epoch, max_epochs, eval_loss, ndcg,
    best_epoch}``, or None — the trainer prints one ``[epoch] n/max …`` line after
    each epoch's eval, and ``best`` is the best epoch seen so far (what early
    stopping will keep)."""
    match = _EPOCH_LINE_RE.search(line)
    if not match:
        return None
    return {
        "epoch": int(match.group(1)),
        "max_epochs": int(match.group(2)),
        "eval_loss": _opt_float(match.group(3)),
        "ndcg": _opt_float(match.group(4)),
        "best_epoch": int(match.group(5)),
    }


def parse_epoch_points(text: str) -> list[dict]:
    """Every per-epoch validation line in ``text`` (see ``parse_epoch_line``)."""
    points: list[dict] = []
    for line in text.split("\n"):
        point = parse_epoch_line(line)
        if point:
            points.append(point)
    return points


def parse_saved_path(text: str) -> str | None:
    """The directory the model was actually saved to. With auto-naming the final path
    (…-mnrl-e7) is only known at the end of training, so the SSE ``done`` event reads
    it from this line instead of echoing the requested output_dir."""
    saved = None
    for line in text.split("\n"):
        match = _SAVED_RE.search(line)
        if match:
            saved = match.group(1)
    return saved


def parse_summary(text: str) -> dict | None:
    """``{best_epoch, ran, early_stopped}`` from the trainer's one-line summary —
    best_epoch is the epoch whose weights were saved."""
    match = None
    for line in text.split("\n"):
        match = _SUMMARY_RE.search(line) or match
    if not match:
        return None
    return {
        "best_epoch": int(match.group(1)),
        "ran": int(match.group(2)),
        "early_stopped": match.group(3) == "yes",
    }


def parse_eval_ndcg(text: str) -> tuple[float | None, float | None]:
    """(before, after) nDCG@10 — the baseline eval and the post-fine-tune eval."""
    before = after = None
    section: str | None = None
    for line in text.split("\n"):
        if "baseline eval" in line:
            section = "before"
        elif "after fine-tuning" in line:
            section = "after"
        match = _NDCG_RE.search(line)
        if match:
            if section == "before" and before is None:
                before = float(match.group(1))
            elif section == "after":
                after = float(match.group(1))
    return before, after
