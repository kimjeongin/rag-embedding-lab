"""서빙 벤치의 순수 로직 — 백분위·정확 랭킹·하드웨어 지문·기록.

Qdrant나 모델이 필요한 부분은 여기서 다루지 않는다(그건 실제 실행으로 확인).
여기서 고정하는 건 "숫자를 어떻게 요약하는가"의 규칙이다.
"""
from __future__ import annotations

import json

from rag.servingbench import (
    exact_rankings,
    fingerprint_key,
    hardware_fingerprint,
    latency_stats,
    load_bench_runs,
    append_bench_run,
)


# ── 지연시간 요약 ─────────────────────────────────────────────────────────────

def test_latency_stats_reports_percentiles_not_just_the_mean():
    """꼬리 지연이 사용자 체감을 결정한다 — 평균만으로는 안 보인다."""
    samples = [10.0] * 95 + [1000.0] * 5          # 5%가 100배 느린 전형적 꼬리
    stats = latency_stats(samples)
    assert stats["n"] == 100
    assert stats["p50_ms"] == 10.0
    assert stats["max_ms"] == 1000.0
    assert stats["p95_ms"] > stats["p50_ms"]      # 꼬리가 드러난다
    assert stats["p99_ms"] == 1000.0
    assert stats["mean_ms"] < stats["p99_ms"] / 10  # 평균(59.5)은 그걸 가린다


def test_latency_stats_handles_single_and_empty_samples():
    assert latency_stats([]) == {}
    one = latency_stats([4.2])
    assert one["p50_ms"] == one["p99_ms"] == 4.2


# ── ANN 없는 천장 랭킹 ────────────────────────────────────────────────────────

def test_exact_rankings_orders_by_cosine_similarity():
    doc_ids = ["a", "b", "c"]
    docs = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    ranked = exact_rankings(doc_ids, docs, ["q"], [[1.0, 0.0]], top_k=3)
    assert ranked["q"][0] == "a"        # 완전 일치가 1위
    assert ranked["q"][1] == "c"        # 그 다음이 가까운 것
    assert ranked["q"][2] == "b"


def test_exact_rankings_respects_top_k_and_short_corpora():
    ranked = exact_rankings(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], ["q"], [[1.0, 0.0]], top_k=50)
    assert ranked["q"] == ["a", "b"]    # corpus보다 큰 top_k도 안전


def test_exact_rankings_normalises_so_magnitude_does_not_win():
    """정규화를 빼먹으면 '긴 문서가 무조건 이기는' 고전적 버그가 난다."""
    ranked = exact_rankings(
        ["short", "long"], [[1.0, 0.0], [50.0, 50.0]], ["q"], [[1.0, 0.0]], top_k=2
    )
    assert ranked["q"][0] == "short"


# ── 하드웨어 지문 ─────────────────────────────────────────────────────────────

def test_hardware_fingerprint_records_what_timings_depend_on():
    info = hardware_fingerprint("cpu", max_seq_length=512)
    for key in ("device", "platform", "torch", "max_seq_length"):
        assert key in info
    assert info["max_seq_length"] == 512


def test_fingerprint_key_separates_devices_and_is_stable():
    cpu = hardware_fingerprint("cpu")
    mps = hardware_fingerprint("mps")
    assert fingerprint_key(cpu) == fingerprint_key(hardware_fingerprint("cpu"))
    assert fingerprint_key(cpu) != fingerprint_key(mps)   # 기기가 다르면 비교 불가


# ── 기록 ──────────────────────────────────────────────────────────────────────

def test_bench_runs_round_trip_through_their_own_registry(tmp_path):
    """정확도 레지스트리와 분리 — 기기를 넘나드는 평균을 애초에 못 내게 한다."""
    path = str(tmp_path / "bench.jsonl")
    assert load_bench_runs(path) == []
    append_bench_run({"id": "abc", "label": "x", "hardware_key": "h1"}, path)
    append_bench_run({"id": "def", "label": "y", "hardware_key": "h2"}, path)
    rows = load_bench_runs(path)
    assert [r["id"] for r in rows] == ["abc", "def"]
    assert json.loads(open(path, encoding="utf-8").readline())["label"] == "x"


# ── device 문자열 ─────────────────────────────────────────────────────────────

def test_memory_sampler_recognises_indexed_device_strings():
    """살아있는 모델은 'mps:0'/'cuda:0'을 준다 — 접두사로 매칭하지 않으면
    GPU 측정이 조용히 비어버린다(실제로 한 번 그렇게 놓쳤다)."""
    import torch

    from rag.servingbench import MemorySampler

    if torch.backends.mps.is_available():
        for device in ("mps", "mps:0"):
            assert MemorySampler(device).source is not None, device
    assert MemorySampler("cpu").source is None


def test_hardware_fingerprint_names_the_accelerator_for_indexed_devices():
    import torch

    if torch.backends.mps.is_available():
        for device in ("mps", "mps:0"):
            assert "apple" in hardware_fingerprint(device).get("gpu", "")
