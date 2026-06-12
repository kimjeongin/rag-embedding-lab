"""Server-owned training jobs — runner state machine, auto-eval, persistence, routes.

The rag-train subprocess is faked, so this exercises the whole runner (stdout parsing →
run states → auto-eval → keep_top_k pruning → disk mirror) without torch or a real
training run. The auto-eval flow is faked too — its own logic is covered by the
evalflow/route tests.
"""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag.api import jobs  # noqa: E402


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, lines, exit_code=0):
        self.stdout = _FakeStdout(lines)
        self.returncode = None
        self._exit_code = exit_code

    async def wait(self):
        self.returncode = self._exit_code
        return self._exit_code

    def kill(self):
        self.returncode = -9


def _train_lines(out_dir: str) -> list[bytes]:
    return [
        b"[train] method=full  loss=mnrl  device=cpu  base_model=base\n",
        b"baseline eval: ndcg@10 = 0.8000\n",
        b"{'loss': 0.5, 'epoch': 1.0}\n",
        b"[epoch] 1/2 eval_loss=0.4500 ndcg@10=0.9000 best=1\n",
        b"[epoch] 2/2 eval_loss=0.4400 ndcg@10=0.9100 best=2\n",
        b"[train] summary best_epoch=2 ran=2 early_stopped=no\n",
        f"[train] saved fine-tuned model to {out_dir}\n".encode(),
        b"after fine-tuning: ndcg@10 = 0.9100\n",
    ]


@pytest.fixture(autouse=True)
def _isolated_jobs(tmp_path, monkeypatch):
    """Fresh jobs dir + empty in-memory registry per test."""
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_active_id", None)
    monkeypatch.setattr(jobs, "_loaded", True)  # don't pick up real runs/jobs
    yield


def _fake_eval(metrics=None):
    async def run_eval_flow(embedder, model, *, label="", note=None, **_kw):
        return {
            "run": {"id": "ev123"},
            "metrics": metrics or {"ndcg@10": 0.91, "recall@50": 1.0},
            "n_queries": 5,
            "split": "dev",
        }

    return run_eval_flow


async def test_runner_trains_then_auto_evals(monkeypatch, tmp_path):
    async def fake_exec(*_args, **_kwargs):
        return _FakeProc(_train_lines("outputs/m-mnrl-e2"))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    import rag.evalflow

    monkeypatch.setattr(rag.evalflow, "run_eval_flow", _fake_eval())

    job = jobs.create_job([{"label": "lr=2e-5", "config": {"output_dir": "outputs/m"}}])
    await jobs._run_job(job["id"])

    run = job["runs"][0]
    assert job["status"] == "done"
    assert run["status"] == "evaluated"
    assert run["result"]["output_dir"] == "outputs/m-mnrl-e2"   # auto-named path from stdout
    assert run["result"]["best_epoch"] == 2
    assert [p["epoch"] for p in run["epochs"]] == [1, 2]
    assert all("elapsed" in p for p in run["epochs"])           # ETA material
    assert run["eval"]["metrics"]["ndcg@10"] == 0.91
    assert run["eval"]["run_id"] == "ev123"

    # disk mirror exists and matches the terminal state
    on_disk = json.loads((jobs.jobs_dir() / f"{job['id']}.json").read_text())
    assert on_disk["status"] == "done"
    # raw stdout was captured for post-mortems
    assert (jobs.jobs_dir() / job["id"] / "run-0.log").exists()


async def test_sweep_runs_sequentially_and_failure_is_isolated(monkeypatch):
    calls = []

    async def fake_exec(*_args, env=None, **_kwargs):
        calls.append(env["TRAIN_LR"])
        if len(calls) == 1:  # first run dies (e.g. OOM) — the sweep must continue
            return _FakeProc([b"RuntimeError: MPS backend out of memory\n"], exit_code=1)
        return _FakeProc(_train_lines(f"outputs/m{len(calls)}-mnrl-e2"))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    import rag.evalflow

    monkeypatch.setattr(rag.evalflow, "run_eval_flow", _fake_eval())

    job = jobs.create_job(
        [
            {"label": "lr=1e-3", "config": {"learning_rate": 1e-3}},
            {"label": "lr=2e-5", "config": {"learning_rate": 2e-5}},
        ]
    )
    assert job["kind"] == "sweep"
    await jobs._run_job(job["id"])

    first, second = job["runs"]
    assert first["status"] == "failed"
    assert "메모리 부족" in (first["hint"] or "")               # actionable Korean hint
    assert second["status"] == "evaluated"                      # sweep survived the failure
    assert calls == ["0.001", "2e-05"]                          # sequential, in order
    assert job["status"] == "done"


async def test_interrupted_jobs_are_marked_on_reload(monkeypatch, tmp_path):
    job = jobs.create_job([{"label": "", "config": {}}])
    job["status"] = "running"
    job["runs"][0]["status"] = "running"
    jobs._persist(job)

    # simulate a fresh process: empty memory, not yet loaded
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_loaded", False)

    reloaded = jobs.get_job(job["id"])
    assert reloaded["status"] == "interrupted"                  # the subprocess died with the server
    assert reloaded["runs"][0]["status"] == "interrupted"


def test_routes_create_poll_and_409_on_concurrent(monkeypatch):
    """Route-level contract: POST starts a background job, GET polls it, second POST 409s."""
    started: list[str] = []
    monkeypatch.setattr(jobs, "start_job", lambda job_id: started.append(job_id))

    from rag.api.app import create_app

    with TestClient(create_app()) as client:
        resp = client.post("/api/jobs", json={"runs": [{"label": "", "config": {"epochs": 2}}]})
        assert resp.status_code == 200
        state = resp.json()
        assert state["kind"] == "train"
        assert started == [state["id"]]                          # runner was launched
        assert state["runs"][0]["config"]["epochs"] == 2

        polled = client.get(f"/api/jobs/{state['id']}")
        assert polled.status_code == 200
        assert polled.json()["status"] == "pending"

        listed = client.get("/api/jobs").json()
        assert [j["id"] for j in listed["jobs"]] == [state["id"]]

        monkeypatch.setattr(jobs, "_active_id", state["id"])     # pretend it's running
        second = client.post("/api/jobs", json={"runs": [{"label": "", "config": {}}]})
        assert second.status_code == 409

        assert client.get("/api/jobs/nope").status_code == 404


def test_stop_and_skip_require_active_job():
    job = jobs.create_job([{"label": "", "config": {}}])
    assert jobs.request_stop(job["id"]) is False                 # not running → no-op
    assert jobs.request_skip(job["id"]) is False


async def test_keep_top_k_prunes_losing_models(monkeypatch, tmp_path):
    """After a sweep, only the top-k model folders survive; losers keep their eval
    records but lose their weights (model_deleted flag)."""
    out_root = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)                                  # outputs/ guard rail base
    scores = iter([0.80, 0.95])

    n = 0

    async def fake_exec(*_args, **_kwargs):
        nonlocal n
        n += 1
        path = out_root / f"m{n}"
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        return _FakeProc(_train_lines(f"outputs/m{n}"))

    async def fake_eval(embedder, model, *, label="", note=None, **_kw):
        return {"run": {"id": f"ev{model}"}, "metrics": {"ndcg@10": next(scores)}, "n_queries": 5, "split": "dev"}

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    import rag.evalflow

    monkeypatch.setattr(rag.evalflow, "run_eval_flow", fake_eval)

    job = jobs.create_job(
        [{"label": "a", "config": {}}, {"label": "b", "config": {}}],
        keep_top_k=1,
    )
    await jobs._run_job(job["id"])

    assert not (out_root / "m1").exists()                        # loser pruned (0.80)
    assert (out_root / "m2").exists()                            # winner kept (0.95)
    assert job["runs"][0]["model_deleted"] is True
    assert job["runs"][1]["model_deleted"] is False
