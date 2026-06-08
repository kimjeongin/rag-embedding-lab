"""Parse a training run's stdout into structured progress — framework-free.

``rag-train`` (a subprocess) streams the HF Trainer's logs: per-step ``{'loss': …,
'epoch': …}`` dicts and a baseline/after ``ndcg@10 = …`` line on either side of the
fine-tune. Both the Gradio UI (live loss curve + before/after KPI) and the HTTP API's
SSE stream need to turn that text into points and numbers, so the regex lives here
once — stdlib ``re`` only, no pandas/gradio/fastapi.

Tolerant of transformers' quoted (``'loss': '0.39'``) and unquoted (``'loss': 0.39``)
forms, and of tqdm's ``\\r`` progress redraws.
"""
from __future__ import annotations

import re

_LOSS_RE = re.compile(r"'loss':\s*'?([0-9.eE+-]+)")
_EPOCH_RE = re.compile(r"'epoch':\s*'?([0-9.eE+-]+)")
_NDCG_RE = re.compile(r"ndcg@10\s*=\s*([0-9.]+)")


def clean_tqdm(text: str) -> str:
    """Collapse tqdm carriage-return redraws so each line reads as its final state."""
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def parse_loss_points(text: str) -> list[dict]:
    """Every logged training step as ``{step, epoch, loss}`` (step is 1-based order)."""
    points: list[dict] = []
    for line in text.split("\n"):
        loss = _LOSS_RE.search(line)
        if loss:
            epoch = _EPOCH_RE.search(line)
            step = len(points) + 1
            points.append(
                {
                    "step": step,
                    "epoch": float(epoch.group(1)) if epoch else float(step),
                    "loss": float(loss.group(1)),
                }
            )
    return points


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
