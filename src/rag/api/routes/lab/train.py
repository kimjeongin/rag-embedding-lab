"""POST /api/train — fine-tune the embedder, streamed over Server-Sent Events.

Training is long and chatty, so we run ``rag-train`` as a subprocess (which also keeps
torch out of the API process) and forward its stdout to the browser as it arrives,
parsing the HF Trainer log into structured events the React Train screen renders live:

    event: start    data: {"cmd": "rag-train base=… epochs=…"}
    event: log      data: {"line": "…"}              # one cleaned stdout line
    event: loss     data: {"step": 1, "epoch": 1.0, "loss": 0.42}
    event: metrics  data: {"before": 0.81, "after": 0.93}   # baseline / post nDCG@10
    event: done     data: {"exit_code": 0, "output_dir": "outputs/embedding-ft"}
    event: error    data: {"detail": "…"}            # if the subprocess can't start

Native SSE via ``StreamingResponse`` (no extra dependency). The subprocess is read with
``asyncio`` so the event loop is never blocked.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag import trainlog
from rag.api.schemas.lab import TrainRequest

router = APIRouter()

# Run rag-train's main() in a fresh, unbuffered interpreter (subprocess isolates torch).
_ARGV = [sys.executable, "-u", "-c", "from rag.cli.train import main; main()"]

# One training run at a time: two trainers saving into the same output_dir would corrupt
# the model artifact. Held for the whole stream; the route checks it for a clean 409.
_run_lock = asyncio.Lock()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _train_env(req: TrainRequest) -> dict[str, str]:
    """The TRAIN_* overrides rag.training.config.TrainingConfig.from_env() reads."""
    return {
        **os.environ,
        "TRAIN_BASE_MODEL": req.base_model,
        "TRAIN_OUTPUT_DIR": req.output_dir,
        "TRAIN_EPOCHS": str(req.epochs),
        "TRAIN_BATCH_SIZE": str(req.batch_size),
        "TRAIN_LR": str(req.learning_rate),
        "TRAIN_DEVICE": req.device,
        "TRAIN_METHOD": req.method,
        "TRAIN_LORA_R": str(req.lora_r),
        "TRAIN_LORA_ALPHA": str(req.lora_alpha),
        "TRAIN_LORA_DROPOUT": str(req.lora_dropout),
    }


async def _stream(req: TrainRequest):
    if _run_lock.locked():  # raced past the route check — refuse inside the stream too
        yield _sse("error", {"detail": "학습이 이미 실행 중입니다 — 중단하거나 완료된 뒤 다시 시작하세요"})
        return

    async with _run_lock:
        lora = f" r={req.lora_r} alpha={req.lora_alpha}" if req.method == "lora" else ""
        cmd = (
            f"rag-train method={req.method}{lora} base={req.base_model} epochs={req.epochs} "
            f"batch={req.batch_size} lr={req.learning_rate} device={req.device or 'auto'} → {req.output_dir}"
        )
        yield _sse("start", {"cmd": cmd})

        try:
            proc = await asyncio.create_subprocess_exec(
                *_ARGV,
                env=_train_env(req),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024,  # tolerate a long tqdm line before its newline
            )
        except Exception as exc:  # noqa: BLE001 — report a failed spawn instead of 500ing
            yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
            return

        try:
            accumulated: list[str] = []
            emitted_loss = 0
            last_metrics: tuple[float | None, float | None] = (None, None)
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                accumulated.append(line)
                yield _sse("log", {"line": trainlog.clean_tqdm(line).rstrip("\n")})

                text = trainlog.clean_tqdm("".join(accumulated))
                points = trainlog.parse_loss_points(text)
                while emitted_loss < len(points):       # emit only newly-seen loss points
                    yield _sse("loss", points[emitted_loss])
                    emitted_loss += 1
                metrics = trainlog.parse_eval_ndcg(text)
                if metrics != last_metrics:
                    last_metrics = metrics
                    yield _sse("metrics", {"before": metrics[0], "after": metrics[1]})

            exit_code = await proc.wait()
            yield _sse("done", {"exit_code": exit_code, "output_dir": req.output_dir})
        finally:
            # A client disconnect (stop button, closed tab) cancels this generator;
            # without the kill the orphaned trainer blocks forever once its stdout
            # pipe fills, holding the device. Disconnect therefore MEANS stop.
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()


@router.post("/train")
async def train(req: TrainRequest) -> StreamingResponse:
    if _run_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="학습이 이미 실행 중입니다 — 진행 중인 학습을 중단한 뒤 다시 시작하세요",
        )
    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # don't let proxies buffer the stream
        },
    )
