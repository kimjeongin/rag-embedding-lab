"""Saved-model housekeeping + the handoff package — outputs/ as a managed shelf.

Every trained model is ~1GB, auto-naming means they accumulate, and the registry
refers to them by path — so the lab needs one place that can answer "what models do I
have, what recipe made each, how big are they, which scores are theirs", delete the
losers safely, and package the winner for the serving team.

The handoff package is the lab's finish line: the lab does NOT deploy. Production
swaps the dense model inside its existing hybrid(BM25+dense)+rerank pipeline, so what
it needs from us is the exact embedding contract (instruction template, document
template, pooling, normalization), sample vectors to verify parity (cosine ≥ 0.999),
and the recipe/scores for the record. Written to HANDOFF.md (humans) + handoff.json
(machines) inside the model dir.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from rag import runs as registry
from rag.config import Settings
from rag.core.formatting import format_document, format_query

HANDOFF_MARKER = "runs/handoff.json"

# Fixed sample texts → deterministic vectors per model. Korean+English on purpose:
# internal-site queries are short, mixed-language, and nothing like LLM prose.
_SAMPLE_QUERIES = ["vpn 안됨", "연차 신청 방법", "사내 wifi 비밀번호 변경"]
_SAMPLE_DOCS = [
    ("VPN 연결 가이드", "사내 VPN 설정 방법과 자주 발생하는 연결 오류 해결 절차를 안내합니다."),
    ("휴가 신청 포털", "연차·반차·병가 신청 절차와 결재선 설정 방법을 설명합니다."),
]


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _read_json(path: Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _dim_of(path: Path) -> int | None:
    pooling = _read_json(path / "1_Pooling" / "config.json") or {}
    dim = pooling.get("embedding_dimension")
    return int(dim) if isinstance(dim, (int, float)) else None


def _eval_summary(record: dict) -> dict:
    return {
        "run_id": record.get("id"),
        "created_at": record.get("created_at"),
        "label": record.get("label"),
        "split": record.get("split"),
        "n_queries": record.get("n_queries"),
        "metrics": record.get("metrics") or {},
    }


def model_detail(path: str, runs: list[dict] | None = None) -> dict:
    """One saved model joined with its recipe and its eval records."""
    p = Path(path)
    runs = registry.load_runs() if runs is None else runs
    mine = [r for r in runs if r.get("model") == path]
    dev = [r for r in mine if r.get("split") in (None, "dev", "test")]
    final = [r for r in mine if r.get("split") == "final"]
    best_dev = max(dev, key=lambda r: (r.get("metrics") or {}).get("ndcg@10", -1.0), default=None)
    return {
        "path": path,
        "size_bytes": _dir_size(p) if p.exists() else 0,
        "dim": _dim_of(p),
        "created_at": (
            datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            if p.exists() else None
        ),
        "meta": _read_json(p / "train_meta.json"),
        "eval_dev": _eval_summary(best_dev) if best_dev else None,
        "eval_final": _eval_summary(final[0]) if final else None,  # newest-first registry
        "handed_off": (p / "HANDOFF.md").exists(),
    }


def list_detail(model_paths: list[str]) -> dict:
    """All saved models with details + the shelf's total disk usage."""
    runs = registry.load_runs()
    models = [model_detail(path, runs) for path in model_paths]
    return {"models": models, "disk_total_bytes": sum(m["size_bytes"] for m in models)}


def delete_model(path: str) -> None:
    """rm -rf one model dir — guarded to outputs/ only. Eval records survive (the
    numbers outlive the weights, same trade keep_top_k makes)."""
    resolved = Path(path).resolve()
    outputs_root = Path("outputs").resolve()
    if outputs_root not in resolved.parents:
        raise ValueError("outputs/ 아래의 모델만 삭제할 수 있습니다")
    if not resolved.is_dir():
        raise ValueError("해당 모델 폴더가 없습니다")
    shutil.rmtree(resolved)


def handed_off_model() -> dict | None:
    """The latest delivery marker ({model, at}) — what the context bar shows."""
    return _read_json(Path(HANDOFF_MARKER))


def build_handoff(path: str) -> dict:
    """Assemble + write the handoff package for one model (blocking — torch encode).

    Returns {"handoff": dict, "markdown": str}. Requires the training stack.
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError("해당 모델 폴더가 없습니다")

    from sentence_transformers import SentenceTransformer

    settings = Settings.from_env()
    instruction = settings.query_instruction
    model = SentenceTransformer(path)

    query_texts = [format_query(q, instruction) for q in _SAMPLE_QUERIES]
    doc_texts = [format_document(t, c) for t, c in _SAMPLE_DOCS]
    vectors = model.encode(query_texts + doc_texts, normalize_embeddings=True)

    # Encoding speed on THIS machine — a relative reference for the serving team's
    # capacity math, not a production latency promise.
    bench_docs = [format_document(f"문서 {i}", "사내 시스템 사용 안내 문서입니다. " * 8) for i in range(32)]
    started = time.perf_counter()
    model.encode(bench_docs, normalize_embeddings=True)
    docs_per_sec = round(32 / max(time.perf_counter() - started, 1e-9), 1)

    detail = model_detail(path)
    samples = [
        {"kind": "query", "text": raw, "input": formatted, "vector": [round(float(v), 6) for v in vec]}
        for raw, formatted, vec in zip(_SAMPLE_QUERIES, query_texts, vectors[: len(query_texts)])
    ] + [
        {"kind": "document", "text": f"{t} — {c}", "input": formatted, "vector": [round(float(v), 6) for v in vec]}
        for (t, c), formatted, vec in zip(_SAMPLE_DOCS, doc_texts, vectors[len(query_texts):])
    ]

    handoff = {
        "model": path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dim": detail["dim"],
        "size_bytes": detail["size_bytes"],
        "recipe": detail["meta"],
        "eval_dev": detail["eval_dev"],
        "eval_final": detail["eval_final"],
        "contract": {
            "query_template": "Instruct: {instruction}\nQuery: {query}",
            "instruction": instruction,
            "document_template": "{title}\n\n{content}",
            "pooling": (_read_json(p / "1_Pooling" / "config.json") or {}).get("pooling_mode"),
            "normalize": True,
            "similarity": "cosine",
        },
        "encode_speed": {"docs_per_sec": docs_per_sec, "batch": 32, "note": "랩 장비 기준 참고치"},
        "samples": samples,
        "checklist": [
            "쿼리/문서 포맷이 contract와 글자 단위로 일치하는지 확인 (instruction 문자열, 개행 포함)",
            "samples의 텍스트를 서빙 파이프라인으로 임베딩해 vector와 cosine ≥ 0.999 확인",
            "모델이 바뀌면 문서 벡터가 전부 바뀝니다 — 전체 재색인 필수",
            "dim이 기존 모델과 다르면 인덱스 매핑(스키마)부터 수정",
            "교체 후 하이브리드 융합 가중치는 그대로 두고 A/B로 시스템 지표 확인",
        ],
    }

    markdown = _render_markdown(handoff)
    (p / "handoff.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (p / "HANDOFF.md").write_text(markdown, encoding="utf-8")

    marker = Path(HANDOFF_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"model": path, "at": handoff["created_at"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"handoff": handoff, "markdown": markdown}


def _fmt_metrics(summary: dict | None) -> str:
    if not summary:
        return "—"
    metrics = summary.get("metrics") or {}
    parts = [f"{k} {v:.4f}" for k, v in metrics.items() if k in ("recall@50", "ndcg@10", "mrr@10")]
    return f"{' · '.join(parts)} (n={summary.get('n_queries')}, {summary.get('split')})"


def _render_markdown(h: dict) -> str:
    recipe = h.get("recipe") or {}
    lines = [
        f"# Handoff — {h['model']}",
        "",
        f"생성: {h['created_at']} · dim {h['dim']} · {round((h['size_bytes'] or 0) / 1e9, 2)}GB",
        "",
        "## 레시피",
        "```json",
        json.dumps(recipe, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 점수",
        f"- dev(선택용): {_fmt_metrics(h.get('eval_dev'))}",
        f"- final(최종 확정): {_fmt_metrics(h.get('eval_final'))}",
        "",
        "## 임베딩 계약 (서빙과 글자 단위로 일치해야 함)",
        f"- 쿼리: `Instruct: {{instruction}}\\nQuery: {{query}}`",
        f"- instruction: `{h['contract']['instruction']}`",
        "- 문서: `{title}\\n\\n{content}` (제목 없으면 본문만)",
        f"- pooling: {h['contract']['pooling']} · L2 normalize · cosine",
        "",
        f"## 인코딩 속도 (참고치): {h['encode_speed']['docs_per_sec']} docs/s (batch {h['encode_speed']['batch']}, 랩 장비)",
        "",
        "## 패리티 검증용 샘플 (handoff.json의 vector와 cosine ≥ 0.999)",
    ]
    for sample in h["samples"]:
        lines.append(f"- [{sample['kind']}] {sample['text']}")
    lines += ["", "## 체크리스트"]
    lines += [f"- [ ] {item}" for item in h["checklist"]]
    return "\n".join(lines) + "\n"
