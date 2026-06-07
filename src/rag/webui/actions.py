"""UI actions — data/logic glue between widgets and the rag.* stack.

Deliberately **gradio-free** (only pandas + httpx + the rag stack): every function takes
plain widget values and returns plain data — strings, DataFrames, pandas Stylers, or
(value, axis) tuples. `app.py` does all the gradio component assembly. Keeping the seam
here means the package imports (and unit-tests) without gradio installed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import re
import shutil
from pathlib import Path

import httpx
import pandas as pd

from rag.config import Settings
from rag.datagen.dummy import generate_dataset
from rag.datagen.eval_corpus import generate as generate_eval_set
from rag.dataset import dataset_paths, load_jsonl, write_jsonl
from rag.evaluation.beir import eval_dir_from_env, write_beir_dataset
from rag.evaluation.retrieval import evaluate
from rag.webui import runs as registry
from rag.webui.jobs import python_call, stream_command

_KPI_METRICS = ("recall@1", "recall@3", "ndcg@10", "mrr@10")

# Inline styles (with !important) for coloured chips. The theme — especially dark mode —
# otherwise recolours text to a light value, leaving it invisible on our light chips.
# An inline `!important` declaration wins over any stylesheet, so this is bulletproof.
_C_WARN = "background:#fff4e5 !important;color:#7a4a00 !important;border:1px solid #ffce85"
_C_INFO = "background:#eef2ff !important;color:#3730a3 !important;border:1px solid #c7d2fe"
_C_OK = "background:#e7f6ec !important;color:#14532d !important;border:1px solid #a7d8b8"
_C_DOWN = "background:#fdecec !important;color:#8a1c1c !important;border:1px solid #f3c2c2"
_BANNER = "display:block;padding:11px 15px;border-radius:12px;line-height:1.55;margin:6px 0;"
_PILL = "display:inline-block;font-size:12.5px;padding:5px 12px;border-radius:999px;margin:2px 6px 2px 0;"
_CHIP = "display:inline-block;font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:999px;margin-top:6px;"


def _banner(colors: str, inner: str) -> str:
    return f"<div style='{_BANNER}{colors}'>{inner}</div>"


def _code(text: str) -> str:
    return (
        f"<code style='background:rgba(0,0,0,.08);color:inherit !important;"
        f"padding:1px 6px;border-radius:5px'>{text}</code>"
    )


# ── status header ─────────────────────────────────────────────────────────────
def ollama_status(url: str | None = None) -> tuple[bool, list[str]]:
    url = url or Settings.from_env().ollama_url
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=2.5)
        resp.raise_for_status()
        return True, [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        return False, []


def device_status() -> str:
    try:
        import torch
    except ImportError:
        return "torch 미설치 (학습 그룹 필요)"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_sample_eval(eval_dir: str) -> bool:
    """True if EVAL_DIR looks like the bundled sample (gold-/distractor- ids)."""
    path = Path(eval_dir) / "corpus.jsonl"
    if not path.exists():
        return False
    try:
        first = next(load_jsonl(str(path)))
    except (StopIteration, FileNotFoundError):
        return False
    return str(first.get("_id", "")).startswith(("distractor-", "gold-"))


def status_html() -> str:
    settings = Settings.from_env()
    ok, _ = ollama_status(settings.ollama_url)
    eval_dir = eval_dir_from_env()
    sample = is_sample_eval(eval_dir)
    n_runs = len(registry.load_runs())
    pills = [
        f"<span style='{_PILL}{_C_OK if ok else _C_WARN}'>Ollama {'🟢' if ok else '🔴'}</span>",
        f"<span style='{_PILL}{_C_INFO}'>device: {device_status()}</span>",
        f"<span style='{_PILL}{_C_WARN if sample else _C_INFO}'>eval: {eval_dir}{' (샘플)' if sample else ''}</span>",
        f"<span style='{_PILL}{_C_INFO}'>runs: {n_runs}</span>",
    ]
    return "<div class='statusbar'>" + "".join(pills) + "</div>"


def _count(path: str) -> int:
    try:
        return sum(1 for _ in load_jsonl(path))
    except FileNotFoundError:
        return 0


def eval_header_html() -> str:
    """Evaluate-tab banner: which eval set is used (binding) + a sample-data warning."""
    eval_dir = eval_dir_from_env()
    binding = (
        f"📂 평가셋 {_code(eval_dir)} · corpus {_count(f'{eval_dir}/corpus.jsonl')} · "
        f"queries {_count(f'{eval_dir}/queries.jsonl')} "
        f"<span style='opacity:.75'>(① 데이터 탭의 ‘평가 데이터’)</span>"
    )
    if is_sample_eval(eval_dir):
        return _banner(
            _C_WARN,
            f"{binding}<br>⚠️ <b>샘플 데이터</b>라 distractor가 쉬워 점수가 높게 나옵니다 — "
            "실제 측정은 사내 데이터를 같은 형식으로 넣으세요 (docs/evaluation.md).",
        )
    return _banner(_C_INFO, binding)


# ── model discovery ───────────────────────────────────────────────────────────
def list_st_models() -> list[str]:
    """Sub-dirs of outputs/ that look like a saved model."""
    root = Path("outputs")
    if not root.exists():
        return []
    return [
        str(p)
        for p in sorted(root.iterdir())
        if p.is_dir() and ((p / "config.json").exists() or (p / "modules.json").exists())
    ]


def list_models(embedder: str, ollama_url: str) -> list[str]:
    if embedder == "ollama":
        return ollama_status(ollama_url)[1]
    return list_st_models()


# ── Data tab ──────────────────────────────────────────────────────────────────
def _pairs_frame(path: str, limit: int | None = None, with_content: bool = False) -> pd.DataFrame:
    records = list(load_jsonl(path)) if Path(path).exists() else []
    if limit:
        records = records[:limit]
    rows = []
    for r in records:
        pos = r.get("positive", {})
        row = {"query": r.get("query"), "정답 제목": pos.get("title")}
        if with_content:
            row["정답 내용"] = pos.get("content", "")
        rows.append(row)
    cols = ["query", "정답 제목"] + (["정답 내용"] if with_content else [])
    return pd.DataFrame(rows, columns=cols)


def _corpus_frame(eval_dir: str, limit: int | None = None, truncate: int | None = None) -> pd.DataFrame:
    path = f"{eval_dir}/corpus.jsonl"
    records = list(load_jsonl(path)) if Path(path).exists() else []
    if limit:
        records = records[:limit]
    rows = []
    for r in records:
        text = r.get("text", "")
        if truncate and len(text) > truncate:
            text = text[:truncate] + "…"
        rows.append({"_id": r.get("_id"), "title": r.get("title"), "text": text})
    return pd.DataFrame(rows, columns=["_id", "title", "text"])


def full_pairs() -> pd.DataFrame:
    """Every training pair (for the 'view all' modal)."""
    train_file, _ = dataset_paths()
    return _pairs_frame(train_file, with_content=True)


def full_corpus() -> pd.DataFrame:
    """Every eval-corpus doc, untruncated (for the 'view all' modal)."""
    return _corpus_frame(eval_dir_from_env())


def data_overview() -> pd.DataFrame:
    """What data exists right now and which step consumes it."""
    train_file, test_file = dataset_paths()
    eval_dir = eval_dir_from_env()
    rows = [
        {"데이터": "학습쌍 (train)", "개수": _count(train_file), "쓰이는 곳": "② 학습", "파일": train_file},
        {"데이터": "학습쌍 (test)", "개수": _count(test_file), "쓰이는 곳": "② 학습 (검증)", "파일": test_file},
        {"데이터": "평가 corpus", "개수": _count(f"{eval_dir}/corpus.jsonl"), "쓰이는 곳": "③ 평가",
         "파일": f"{eval_dir}/corpus.jsonl"},
        {"데이터": "평가 queries", "개수": _count(f"{eval_dir}/queries.jsonl"), "쓰이는 곳": "③ 평가",
         "파일": f"{eval_dir}/queries.jsonl"},
    ]
    return pd.DataFrame(rows, columns=["데이터", "개수", "쓰이는 곳", "파일"])


def train_data_info() -> str:
    """Train-tab banner: which data this run will train on (binding + counts)."""
    train_file, test_file = dataset_paths()
    return _banner(
        _C_INFO,
        f"📂 학습 데이터 {_code(train_file)} ({_count(train_file)} pairs) · "
        f"검증 {_code(test_file)} ({_count(test_file)}) "
        f"<span style='opacity:.75'>(① 데이터 탭의 ‘학습 데이터’)</span>",
    )


def gen_pairs(method: str, corpus_file: str, gen_model: str, n_queries, hard_negatives) -> tuple[str, pd.DataFrame]:
    train_file, eval_file = dataset_paths()
    if method == "toy":
        train, test = generate_dataset()
    else:
        from rag.datagen.synthetic import generate

        try:
            train, test = asyncio.run(
                generate(corpus_file, gen_model, int(n_queries), int(hard_negatives), Settings.from_env())
            )
        except Exception as exc:  # noqa: BLE001 — surface to the UI
            return f"❌ {type(exc).__name__}: {exc}\n(Ollama 실행 중인가요? '{gen_model}' 받으셨나요?)", pd.DataFrame()
    write_jsonl(train_file, train)
    write_jsonl(eval_file, test)
    return f"✅ 학습쌍 저장: {train_file} ({len(train)}) + {eval_file} ({len(test)})", _pairs_frame(train_file, 8)


def gen_eval_set(n_distractors) -> tuple[str, pd.DataFrame]:
    eval_dir = eval_dir_from_env()
    corpus, queries, qrels = generate_eval_set(n_distractors=int(n_distractors) or None)
    write_beir_dataset(eval_dir, corpus, queries, qrels)
    msg = f"✅ 평가셋 저장: {eval_dir} ({len(corpus)} docs · {len(queries)} queries · {len(qrels)} qrels)"
    return msg, _corpus_frame(eval_dir, 8, truncate=80)


# ── Train tab: dependency readiness (so the user never touches the CLI) ─────────
def training_ready() -> bool:
    """True if the training stack (torch + sentence-transformers) is importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "sentence_transformers"))


def training_status_html() -> str:
    if training_ready():
        return _banner(_C_INFO, "✅ 학습 라이브러리 설치됨 — 바로 학습할 수 있어요.")
    return _banner(
        _C_WARN, "⚠️ 학습용 라이브러리(torch 등)가 아직 없어요. 아래 버튼으로 <b>한 번만</b> 설치하면 됩니다."
    )


def install_training():
    """Stream `uv sync --group ui --group training` — the one-time training-stack install.

    Includes `--group ui` so gradio is kept (a bare `uv sync --group training` would drop
    it and break the running server).
    """
    uv = shutil.which("uv") or "uv"
    header = "$ uv sync --group ui --group training\n(처음이면 torch 등 다운로드로 몇 분 걸릴 수 있어요)\n\n"
    yield header
    for accumulated in stream_command([uv, "sync", "--group", "ui", "--group", "training"]):
        yield header + accumulated


# ── Train tab (parse the streamed log into a loss curve + before/after KPI) ─────
# Tolerant of transformers' quoted ('loss': '0.39') and unquoted ('loss': 0.39) logs.
_LOSS_RE = re.compile(r"'loss':\s*'?([0-9.eE+-]+)")
_EPOCH_RE = re.compile(r"'epoch':\s*'?([0-9.eE+-]+)")
_NDCG_RE = re.compile(r"ndcg@10\s*=\s*([0-9.]+)")


def _clean_tqdm(text: str) -> str:
    """Collapse tqdm carriage-return redraws so the log reads cleanly in a textbox."""
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def _parse_loss(text: str) -> pd.DataFrame:
    rows = []
    for line in text.split("\n"):
        loss = _LOSS_RE.search(line)
        if loss:
            epoch = _EPOCH_RE.search(line)
            rows.append(
                {"step": len(rows) + 1, "epoch": float(epoch.group(1)) if epoch else float(len(rows) + 1),
                 "loss": float(loss.group(1))}
            )
    return pd.DataFrame(rows, columns=["step", "epoch", "loss"])


def _parse_eval(text: str) -> tuple[float | None, float | None]:
    before = after = None
    section = None
    for line in text.split("\n"):
        if "baseline eval" in line:
            section = "before"
        elif "after fine-tuning" in line:
            section = "after"
        m = _NDCG_RE.search(line)
        if m:
            if section == "before" and before is None:
                before = float(m.group(1))
            elif section == "after":
                after = float(m.group(1))
    return before, after


def _train_kpi_html(before: float | None, after: float | None) -> str:
    if before is None and after is None:
        return ""
    cards = []
    if before is not None:
        cards.append(_kpi_card("nDCG@10 (학습 전)", before))
    if after is not None:
        delta = (after - before) if before is not None else None
        cards.append(_kpi_card("nDCG@10 (학습 후)", after, delta))
    return f"<div class='kpi-row'>{''.join(cards)}</div>"


def run_train(base_model, epochs, batch_size, learning_rate, output_dir, device):
    """Stream `rag-train` (subprocess → torch stays out of the UI); yield (log, loss_df, kpi)."""
    env = {
        "TRAIN_BASE_MODEL": str(base_model),
        "TRAIN_EPOCHS": str(int(epochs)),
        "TRAIN_BATCH_SIZE": str(int(batch_size)),
        "TRAIN_LR": str(float(learning_rate)),
        "TRAIN_OUTPUT_DIR": str(output_dir),
        "TRAIN_DEVICE": device or "",
    }
    header = (
        f"$ rag-train  base={base_model}  epochs={epochs}  batch={batch_size}  "
        f"lr={learning_rate}  device={device or 'auto'}  → {output_dir}\n"
        "(needs `uv sync --group training`; the first run downloads the base model)\n\n"
    )
    yield header, pd.DataFrame(columns=["step", "epoch", "loss"]), ""
    for accumulated in stream_command(python_call("rag.cli.train"), env):
        clean = _clean_tqdm(accumulated)
        yield header + clean, _parse_loss(clean), _train_kpi_html(*_parse_eval(clean))


# ── Evaluate tab ──────────────────────────────────────────────────────────────
def _settings(embedder: str, model: str, embed_dim: int, ollama_url: str) -> Settings:
    base = Settings.from_env()
    if embedder == "ollama":
        return Settings(
            embedder="ollama", embed_model=(model or base.embed_model), embed_dim=embed_dim,
            ollama_url=(ollama_url or base.ollama_url), query_instruction=base.query_instruction,
        )
    return Settings(
        embedder="sentence-transformers", st_model=(model or base.st_model), embed_dim=embed_dim,
        query_instruction=base.query_instruction,
    )


def _infer_dim(embedder: str, model: str, ollama_url: str) -> int:
    if embedder == "ollama":
        resp = httpx.post(f"{ollama_url}/api/embed", json={"model": model, "input": "x"}, timeout=30)
        resp.raise_for_status()
        return len(resp.json()["embeddings"][0])
    from sentence_transformers import SentenceTransformer

    return int(SentenceTransformer(model).get_sentence_embedding_dimension())


def _kpi_card(label: str, value: float, delta: float | None = None) -> str:
    chip = ""
    if delta is not None:
        if delta > 1e-6:
            chip = f"<span style='{_CHIP}{_C_OK}'>▲ {delta:+.4f}</span>"
        elif delta < -1e-6:
            chip = f"<span style='{_CHIP}{_C_DOWN}'>▼ {delta:+.4f}</span>"
        else:
            chip = f"<span style='{_CHIP}{_C_INFO}'>±0</span>"
    return f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value:.4f}</div>{chip}</div>"


def _eval_kpi_html(metrics: dict, prior_best: dict) -> str:
    cards = [
        _kpi_card(key, metrics[key], (metrics[key] - prior_best[key]) if prior_best.get(key) is not None else None)
        for key in _KPI_METRICS
        if key in metrics
    ]
    note = (
        f"<span style='{_CHIP}{_C_INFO};margin-top:2px'>▲▼ 기존 best 대비</span>"
        if any(prior_best.values())
        else _banner(_C_INFO, "첫 평가 — 차이(▲▼)는 다음 평가부터 표시됩니다.")
    )
    return f"<div class='kpi-row'>{''.join(cards)}</div>{note}"


def run_eval(embedder, model, ollama_url, eval_dir, label):
    """Evaluate the chosen model, append to the registry, yield (status, kpi_html).

    The embedding dimension is auto-detected from the model, so there's no manual field
    to get wrong.
    """
    eval_dir = (eval_dir or "").strip() or eval_dir_from_env()
    yield f"⏳ 평가 중… ({model} 으로 코퍼스+쿼리 임베딩)", ""
    try:
        dim = _infer_dim(embedder, model, ollama_url)
        settings = _settings(embedder, model, dim, ollama_url)
        prior_best = registry.best_per_metric()
        metrics = asyncio.run(evaluate(settings, eval_dir))
    except Exception as exc:  # noqa: BLE001 — surface to the UI
        yield f"❌ {type(exc).__name__}: {exc}", ""
        return
    if not metrics:
        yield "⚠️ 판정된 쿼리가 없습니다 — qrels/test.tsv 확인", ""
        return
    registry.append_run(label, embedder, settings.active_model, eval_dir, metrics)
    yield f"✅ 완료 — {settings.active_model}  (dim={dim})", _eval_kpi_html(metrics, prior_best)


# ── Compare tab (data only; app.py wraps these into components) ────────────────
def _runs_frame() -> pd.DataFrame:
    # leading 🗑 column = per-row delete handle (clicking it removes that run)
    cols = ["🗑", "label", "model", *registry.METRIC_KEYS, "when"]
    rows = []
    for r in registry.load_runs():
        row = {"🗑": "🗑", "label": r.get("label"), "model": r.get("model"),
               "when": (r.get("created_at") or "")[5:16]}
        for key in registry.METRIC_KEYS:
            value = r.get("metrics", {}).get(key)
            row[key] = round(float(value), 4) if value is not None else None
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _highlight_best(column: pd.Series) -> list[str]:
    """Green chip on the best (max) cell — sets BOTH bg and text colour so it stays
    readable in light *and* dark mode (highlight_max only sets the background)."""
    best = column.max()
    style = "background-color:#d8f0dd; color:#14532d; font-weight:700"
    return [style if value == best else "" for value in column]


def compare_styled():
    """Runs table with the best value per metric highlighted (pandas Styler)."""
    df = _runs_frame()
    if df.empty:
        return df
    metric_cols = [k for k in registry.METRIC_KEYS if k in df.columns]
    return (
        df.style.apply(_highlight_best, subset=metric_cols, axis=0)
        .format({k: "{:.4f}" for k in metric_cols}, na_rep="—")
    )


def compare_all_data() -> tuple[pd.DataFrame, list[float]]:
    """Long-form (metric, run, value) for a grouped bar chart of ALL metrics at once,
    with the y-axis zoomed to the data so small differences stay visible."""
    rows, values = [], []
    for r in registry.load_runs():
        label = r.get("label") or r.get("model")
        for key in registry.METRIC_KEYS:
            v = r.get("metrics", {}).get(key)
            if v is not None:
                rows.append({"metric": key, "run": label, "value": round(float(v), 4)})
                values.append(float(v))
    df = pd.DataFrame(rows, columns=["metric", "run", "value"])
    lo = min(values) if values else 0.0
    return df, [max(0.0, lo - 0.03), 1.0]


def compare_figure():
    """Grouped (side-by-side) bar chart of ALL metrics — one bar per model per metric.

    Uses plotly's barmode='group' (Gradio's native BarPlot stacks colours instead of
    grouping). The y-axis is zoomed to the data so small gaps are visible.
    """
    import plotly.express as px

    df, y_lim = compare_all_data()
    if df.empty:
        fig = px.bar(title="아직 평가 결과가 없어요 — ③ 평가에서 모델을 평가해 보세요")
    else:
        fig = px.bar(
            df, x="metric", y="value", color="run", barmode="group",
            range_y=y_lim, title="지표별 점수 — 모델별 막대 (높을수록 좋음)",
        )
    # transparent background blends into the page in both light and dark mode
    fig.update_layout(
        height=360, margin={"l": 10, "r": 10, "t": 48, "b": 10},
        legend_title_text="model", title_font_size=15,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Pretendard, sans-serif", "color": "#8b93a7", "size": 12},
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)", title_text=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    return fig


def default_model(embedder: str, choices: list[str]) -> str:
    """A sensible default selection when the backend changes (so a stale model isn't kept)."""
    if not choices:
        return ""
    if embedder == "ollama":
        for choice in choices:
            if "embedding" in choice:
                return choice
    return choices[0]


def delete_run_at(row_index: int | None) -> None:
    """Delete the run at the given table row (rows are newest-first, like the table)."""
    if row_index is None:
        return
    runs = registry.load_runs()
    if 0 <= row_index < len(runs):
        registry.delete_run(runs[row_index]["id"])
