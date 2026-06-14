"""Server-owned training jobs — single runs and sweeps share one runner.

Why server-owned: the old POST /api/train tied the training's LIFETIME to the SSE
connection (disconnect = kill). Fine for a demo, fatal for real work — a page refresh
or a laptop sleep destroyed an hour of training. Here the API process owns the job;
browsers just poll its state and can re-attach any time. A sweep is simply a job with
more than one run, executed sequentially (one device), each optionally auto-evaluated
the moment it finishes so the sweep page is a live leaderboard.

State lives in memory (single worker) and is mirrored to runs/jobs/{id}.json on every
transition, so history survives restarts. A job that was mid-flight when the server
died is marked "interrupted" on the next load — its subprocess died with the server.
Raw training output goes to runs/jobs/{id}/run-{idx}.log for post-mortems.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import shutil
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from rag import trainlog
from rag.api.schemas.lab import TrainRequest

DEFAULT_JOBS_DIR = "runs/jobs"

# Run rag-train's main() in a fresh, unbuffered interpreter (subprocess isolates torch
# from the API process; training crashes can't take the server down).
_ARGV = [sys.executable, "-u", "-c", "from rag.cli.train import main; main()"]

# In-memory registry — one uvicorn worker owns all jobs. Disk is a mirror, not a queue.
_jobs: dict[str, dict] = {}
_active_id: str | None = None
_proc: asyncio.subprocess.Process | None = None
_stop = False    # stop the whole job (current run killed, rest skipped)
_skip = False    # kill just the current run, move on to the next
_loaded = False

# Common failure signatures → an actionable hint in the run state (the raw log is
# still there; this is the "what do I do now" line the UI shows first).
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("out of memory", "mps backend out of memory", "cuda out of memory"),
     "메모리 부족입니다 — batch size를 줄이거나 LoRA로 전환해 보세요. batch를 유지하고 싶으면 Cached MNRL이 "
     "메모리를 아껴줍니다. Matryoshka를 켰다면 여러 차원의 backward 그래프를 동시에 들고 있어 batch·LoRA로도 "
     "잘 안 줄어드니, 차원 수를 줄이세요(또는 VRAM이 더 큰 GPU)."),
    (("no module named", "modulenotfounderror"),
     "학습 스택이 설치되어 있지 않습니다 — `uv sync --group training` 후 다시 시도하세요."),
    (("hard negative",),
     "Triplet loss에는 모든 레코드에 hard negative가 필요합니다 — 데이터 탭에서 hard-negative mining을 켜고 재생성하세요."),
    (("connection refused", "connecterror", "11434"),
     "Ollama가 꺼져 있는 것 같습니다 — `ollama serve`가 실행 중인지 확인하세요."),
    (("no space left",),
     "디스크가 가득 찼습니다 — 모델 페이지에서 안 쓰는 모델을 정리하세요 (런당 약 1GB)."),
)


# Cross-run median pruning (sweep-only, opt-in): while a run trains, compare its
# best-so-far validation nDCG against the runs that already finished, at the same
# epoch, and kill it early if it trails the median — the single biggest compute saving
# for an expensive sequential sweep (Optuna's MedianPruner, on the metric we already
# stream). Held off until a few epochs in (early curves are noisy) and until enough
# peers exist to form a median.
_PRUNE_WARMUP = 2       # ≥ this many completed runs before a median means anything
_PRUNE_MIN_EPOCH = 2    # never prune before a run has had this many epochs


def _best_ndcg_at(epochs: list[dict], epoch: int) -> float | None:
    """A run's best val nDCG@10 through ``epoch`` (monotone — the value early stopping
    would keep). None when no epoch up to here reported an nDCG."""
    seen = [e["ndcg"] for e in epochs if e.get("ndcg") is not None and e.get("epoch", 0) <= epoch]
    return max(seen) if seen else None


def _should_prune(current: float | None, peer_bests: list[float], epoch: int) -> bool:
    """True when a run at ``epoch`` should be pruned: its best-so-far trails the median
    of the completed peers'. Pure (no I/O) so it's unit-testable."""
    if epoch < _PRUNE_MIN_EPOCH or current is None or len(peer_bests) < _PRUNE_WARMUP:
        return False
    return current < statistics.median(peer_bests)


def _peer_bests_at(job: dict, current_idx: int, epoch: int) -> list[float]:
    """best-so-far nDCG at ``epoch`` for every COMPLETED run other than this one.
    Completed-only: a run that was itself pruned/failed isn't a fair baseline."""
    out = []
    for r in job["runs"]:
        if r["idx"] == current_idx or r.get("status") not in ("trained", "evaluated"):
            continue
        best = _best_ndcg_at(r.get("epochs") or [], epoch)
        if best is not None:
            out.append(best)
    return out


def jobs_dir() -> Path:
    """Where job state + logs live (JOBS_DIR), defaulting to runs/jobs."""
    return Path(os.getenv("JOBS_DIR", DEFAULT_JOBS_DIR))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _persist(job: dict) -> None:
    """Mirror one job to disk (atomic replace — a crash mid-write must not eat it)."""
    path = jobs_dir() / f"{job['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _ensure_loaded() -> None:
    """Load persisted jobs once per process. Anything still 'running' on disk died
    with the previous server — mark it interrupted instead of pretending."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    base = jobs_dir()
    if not base.exists():
        return
    for path in base.glob("j-*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(job, dict) or job.get("id") in _jobs:
            continue
        changed = False
        if job.get("status") in ("pending", "running"):
            job["status"] = "interrupted"
            changed = True
        for run in job.get("runs", []):
            if run.get("status") in ("pending", "running"):
                run["status"] = "interrupted"
                changed = True
        job["current"] = None
        _jobs[job["id"]] = job
        if changed:
            _persist(job)


# ── public state accessors ───────────────────────────────────────────────────────
def list_jobs() -> list[dict]:
    """All jobs, newest first."""
    _ensure_loaded()
    return sorted(_jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)


def get_job(job_id: str) -> dict | None:
    _ensure_loaded()
    return _jobs.get(job_id)


def active_job_id() -> str | None:
    return _active_id


def delete_job(job_id: str) -> bool:
    """Remove a finished job's record + logs (the trained models stay in outputs/)."""
    _ensure_loaded()
    if job_id == _active_id or job_id not in _jobs:
        return False
    _jobs.pop(job_id)
    with contextlib.suppress(OSError):
        (jobs_dir() / f"{job_id}.json").unlink()
    shutil.rmtree(jobs_dir() / job_id, ignore_errors=True)
    return True


# ── job creation + control ───────────────────────────────────────────────────────
def create_job(
    runs: list[dict],
    auto_eval: bool = True,
    keep_top_k: int | None = None,
    prune: bool = False,
) -> dict:
    """Register a job (not yet started): runs = [{"label": str, "config": dict}]."""
    _ensure_loaded()
    job = {
        "id": f"j-{uuid.uuid4().hex[:8]}",
        "kind": "sweep" if len(runs) > 1 else "train",
        "status": "pending",
        "created_at": _now(),
        "auto_eval": auto_eval,
        "keep_top_k": keep_top_k,
        "prune": prune and len(runs) > 1,   # pruning needs peers — meaningless for a single run
        "current": None,
        "error": None,
        "runs": [
            {
                "idx": i,
                "label": (r.get("label") or "").strip(),
                "status": "pending",
                "config": r["config"],
                "loss": [],
                "epochs": [],
                "result": None,
                "eval": None,
                "error": None,
                "hint": None,
                "started_at": None,
                "finished_at": None,
                "model_deleted": False,
            }
            for i, r in enumerate(runs)
        ],
    }
    _jobs[job["id"]] = job
    _persist(job)
    return job


def start_job(job_id: str) -> None:
    """Launch the runner as a background task — the caller's request returns at once."""
    asyncio.get_running_loop().create_task(_run_job(job_id))


def request_stop(job_id: str) -> bool:
    """Stop the whole job: kill the current run, skip the rest."""
    global _stop
    if _active_id != job_id:
        return False
    _stop = True
    _kill_current()
    return True


def request_skip(job_id: str) -> bool:
    """Kill just the current run and move on to the next one."""
    global _skip
    if _active_id != job_id:
        return False
    _skip = True
    _kill_current()
    return True


def _kill_current() -> None:
    if _proc is not None and _proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            _proc.kill()


# ── the runner ───────────────────────────────────────────────────────────────────
async def _run_job(job_id: str) -> None:
    global _active_id, _stop
    job = _jobs[job_id]
    _active_id = job_id
    _stop = False
    job["status"] = "running"
    _persist(job)
    try:
        for run in job["runs"]:
            if _stop:
                run["status"] = "skipped"
                continue
            job["current"] = run["idx"]
            _persist(job)
            await _run_one(job, run)
            _persist(job)
        if job.get("keep_top_k") and not _stop:
            _cull_loser_models(job)
        job["status"] = "stopped" if _stop else "done"
    except Exception as exc:  # noqa: BLE001 — a runner bug must not leave the job dangling
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["current"] = None
        _active_id = None
        _persist(job)


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
        "TRAIN_LOSS": req.loss,
        "TRAIN_MATRYOSHKA": "1" if req.matryoshka else "0",
        "TRAIN_MATRYOSHKA_DIMS": ",".join(str(d) for d in req.matryoshka_dims),
        "TRAIN_DROPOUT": "" if req.dropout is None else str(req.dropout),
        "TRAIN_PATIENCE": str(req.early_stop_patience),
        "TRAIN_MONITOR": req.early_stop_metric,
        "TRAIN_AUTO_NAME": "1" if req.auto_name else "0",
        "TRAIN_SEED": str(req.seed),
        "TRAIN_NOTE": req.note,
        "TRAIN_METHOD": req.method,
        "TRAIN_LORA_R": str(req.lora_r),
        "TRAIN_LORA_ALPHA": str(req.lora_alpha),
        "TRAIN_LORA_DROPOUT": str(req.lora_dropout),
        "TRAIN_LORA_TARGET": req.lora_target,
    }


async def _run_one(job: dict, run: dict) -> None:
    global _proc, _skip
    _skip = False
    req = TrainRequest(**run["config"])
    run["status"] = "running"
    run["started_at"] = _now()

    log_path = jobs_dir() / job["id"] / f"run-{run['idx']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = await asyncio.create_subprocess_exec(
            *_ARGV,
            env=_train_env(req),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=1024 * 1024,  # tolerate a long tqdm line before its newline
        )
    except Exception as exc:  # noqa: BLE001 — report a failed spawn as a failed run
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["finished_at"] = _now()
        return

    _proc = proc
    accumulated: list[str] = []
    seen_loss = seen_epochs = 0
    pruned = False
    started = time.monotonic()
    try:
        with log_path.open("a", encoding="utf-8") as log_f:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                accumulated.append(line)
                log_f.write(trainlog.clean_tqdm(line))

                text = trainlog.clean_tqdm("".join(accumulated))
                points = trainlog.parse_loss_points(text)
                if seen_loss < len(points):
                    run["loss"].extend(points[seen_loss:])
                    seen_loss = len(points)
                epochs = trainlog.parse_epoch_points(text)
                if seen_epochs < len(epochs):
                    for point in epochs[seen_epochs:]:
                        point["elapsed"] = round(time.monotonic() - started, 1)
                        run["epochs"].append(point)
                    seen_epochs = len(epochs)
                    _persist(job)  # epoch boundary — cheap (~once a minute), keeps disk current

                    # median pruning — judged at the epoch boundary on the metric we
                    # just streamed; kill a clearly-trailing run so the sweep moves on
                    if job.get("prune") and not pruned and not _stop and not _skip:
                        epoch = run["epochs"][-1]["epoch"]
                        current = _best_ndcg_at(run["epochs"], epoch)
                        peers = _peer_bests_at(job, run["idx"], epoch)
                        if _should_prune(current, peers, epoch):
                            pruned = True
                            run["hint"] = (
                                f"중간 검증 nDCG가 완료된 런들의 중앙값을 밑돌아 epoch {epoch}에서 "
                                "조기 종료(median pruning) — keep_top_k처럼 나쁜 후보에 시간을 안 씁니다"
                            )
                            _kill_current()
        exit_code = await proc.wait()
    finally:
        _proc = None

    run["finished_at"] = _now()
    text = trainlog.clean_tqdm("".join(accumulated))

    if _stop or _skip:
        run["status"] = "stopped" if _stop else "skipped"
        return
    if pruned:
        run["status"] = "pruned"   # killed by median pruning — no model saved, no eval
        return
    if exit_code != 0:
        tail = "\n".join(text.split("\n")[-25:])
        run["status"] = "failed"
        run["error"] = f"학습 프로세스 종료 (exit {exit_code})"
        run["hint"] = _hint_for(tail)
        return

    before, after = trainlog.parse_eval_ndcg(text)
    run["result"] = {
        "output_dir": trainlog.parse_saved_path(text) or req.output_dir,
        "ndcg_before": before,
        "ndcg_after": after,
        **(trainlog.parse_summary(text) or {}),
    }
    run["status"] = "trained"
    _persist(job)

    if job["auto_eval"]:
        await _auto_eval(run)


async def _auto_eval(run: dict) -> None:
    """Evaluate the freshly trained model on the dev split and record the run —
    closing the train→eval loop without a human click. Failure is non-fatal: the
    model is saved either way and can be evaluated manually."""
    from rag.evalflow import run_eval_flow

    model = run["result"]["output_dir"]
    try:
        result = await run_eval_flow(
            "sentence-transformers",
            model,
            label=run["label"] or model,
            note=(run["config"].get("note") or None),
        )
        run["eval"] = {
            "run_id": result["run"]["id"],
            "metrics": result["metrics"],
            "n_queries": result["n_queries"],
            "split": result["split"],
        }
        run["status"] = "evaluated"
    except Exception as exc:  # noqa: BLE001 — eval failure must not fail the training run
        run["error"] = f"자동 평가 실패: {type(exc).__name__}: {exc}"
        run["hint"] = "모델은 저장되어 있습니다 — 평가 탭에서 수동으로 평가할 수 있어요."
    finally:
        _release_torch_memory()


def _release_torch_memory() -> None:
    """Each auto-eval loads a ~1GB model into this process; without an explicit
    release a long sweep would accumulate them."""
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cull_loser_models(job: dict) -> None:
    """keep_top_k: after the sweep, delete the model folders of the losing runs
    (~1GB each). Their eval records stay in the registry — the numbers survive,
    the weights don't. Only the winners' folders remain. (Distinct from median
    pruning, which stops a run mid-training; this culls finished models on disk.)"""
    evaluated = [r for r in job["runs"] if r.get("eval") and r.get("result")]
    ranked = sorted(
        evaluated, key=lambda r: r["eval"]["metrics"].get("ndcg@10", 0.0), reverse=True
    )
    keep = {r["result"]["output_dir"] for r in ranked[: job["keep_top_k"]]}
    outputs_root = Path("outputs").resolve()
    for run in evaluated:
        out = run["result"]["output_dir"]
        if out in keep:
            continue
        path = Path(out).resolve()
        # guard rails: never delete anything outside outputs/
        if outputs_root in path.parents and path.exists():
            shutil.rmtree(path, ignore_errors=True)
            run["model_deleted"] = True


def _hint_for(tail: str) -> str | None:
    lowered = tail.lower()
    for needles, hint in _HINTS:
        if any(needle in lowered for needle in needles):
            return hint
    return None
