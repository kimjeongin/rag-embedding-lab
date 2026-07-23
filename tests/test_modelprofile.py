"""모델 입력 포맷 프로파일 — 조용히 틀리는 실패를 테스트로 막는다.

포맷이 틀려도 예외는 안 난다: 벡터는 나오고 검색도 되고 점수만 나빠진다. 그래서
해석 규칙(이름 / train_meta.json / 명시 override)과 각 프로파일이 만드는 정확한
문자열을 여기서 고정한다.
"""
from __future__ import annotations

import json

import pytest

from rag.core.formatting import (
    DEFAULT_QUERY_INSTRUCTION,
    NEMOTRON3,
    PLAIN,
    QWEN3,
    format_document,
    format_query,
    profile_for_name,
)
from rag.modelprofile import UnknownProfile, resolve_profile


# ── 포맷 문자열 ────────────────────────────────────────────────────────────────

def test_qwen_format_is_unchanged_from_before_profiles_existed():
    """기존에 기록된 런이 그대로 재현되어야 한다 — 기본 포맷은 바뀌면 안 된다."""
    assert format_query("연차 신청", "TASK") == "Instruct: TASK\nQuery: 연차 신청"
    assert format_document("제목", "본문") == "제목\n\n본문"
    assert format_document(None, "본문") == "본문"


def test_nemotron_uses_literal_prefixes_and_ignores_the_instruction():
    assert format_query("연차 신청", "TASK", NEMOTRON3) == "query: 연차 신청"
    assert format_document("제목", "본문", NEMOTRON3) == "passage: 제목\n\n본문"
    assert format_document(None, "본문", NEMOTRON3) == "passage: 본문"
    assert NEMOTRON3.uses_instruction is False


def test_plain_profile_adds_nothing_to_the_query():
    assert format_query("연차 신청", DEFAULT_QUERY_INSTRUCTION, PLAIN) == "연차 신청"


def test_instruction_only_reaches_models_whose_profile_uses_it():
    """uses_instruction=False인 프로파일은 instruction을 흘리지 않아야 한다."""
    for profile in (NEMOTRON3, PLAIN):
        assert "SECRET" not in format_query("q", "SECRET", profile)
    assert "SECRET" in format_query("q", "SECRET", QWEN3)


# ── 이름 기반 해석 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("model", "expected"), [
    ("Qwen/Qwen3-Embedding-0.6B", QWEN3),
    ("qwen3-embedding:0.6b", QWEN3),
    ("nvidia/Nemotron-3-Embed-1B-BF16", NEMOTRON3),
    ("nvidia/Nemotron-3-Embed-8B-BF16", NEMOTRON3),
])
def test_known_model_names_resolve_to_their_profile(model, expected):
    assert profile_for_name(model) is expected
    assert resolve_profile(model) is expected


def test_unknown_name_has_no_implied_profile():
    """None을 돌려줘야 호출자가 '기본값+경고'와 '에러'를 선택할 수 있다."""
    assert profile_for_name("intfloat/multilingual-e5-large") is None


def test_unknown_model_falls_back_to_the_default_with_a_warning(capsys):
    assert resolve_profile("some/unknown-model") is QWEN3
    assert "MODEL_PROFILE" in capsys.readouterr().out    # 조용히 넘어가지 않는다


# ── 파인튜닝 산출물(경로) 해석 ─────────────────────────────────────────────────

def _write_meta(tmp_path, **meta):
    (tmp_path / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(tmp_path)


def test_finetuned_dir_inherits_the_profile_of_its_base_model(tmp_path):
    """산출물은 이름이 아니라 경로다 — base_model 기록으로만 포맷을 알 수 있다."""
    model = _write_meta(tmp_path, base_model="nvidia/Nemotron-3-Embed-1B-BF16")
    assert resolve_profile(model) is NEMOTRON3


def test_recorded_profile_wins_over_the_base_model_name(tmp_path):
    model = _write_meta(tmp_path, base_model="Qwen/Qwen3-Embedding-0.6B",
                        model_profile="plain")
    assert resolve_profile(model) is PLAIN


def test_unreadable_meta_falls_through_to_name_matching(tmp_path):
    (tmp_path / "train_meta.json").write_text("{ not json", encoding="utf-8")
    (nemo := tmp_path / "nemotron-ft").mkdir()
    (nemo / "train_meta.json").write_text("{ not json", encoding="utf-8")
    assert resolve_profile(str(nemo)) is NEMOTRON3      # 경로 이름으로 회수


# ── 명시 override ─────────────────────────────────────────────────────────────

def test_explicit_override_beats_everything(tmp_path):
    model = _write_meta(tmp_path, base_model="Qwen/Qwen3-Embedding-0.6B")
    assert resolve_profile(model, "nemotron3") is NEMOTRON3


def test_unknown_override_is_a_hard_error():
    """override는 사용자가 명시한 의도다 — 조용히 무시하면 안 된다."""
    with pytest.raises(UnknownProfile):
        resolve_profile("Qwen/Qwen3-Embedding-0.6B", "qwen4")


# ── 평가 경로가 프로파일 override를 잃지 않는지 ───────────────────────────────

def test_eval_settings_carry_the_profile_override(monkeypatch):
    """학습/서빙은 MODEL_PROFILE을 따르는데 평가만 무시하면 비교가 오염된다."""
    from rag.lab import build_eval_settings

    monkeypatch.setenv("MODEL_PROFILE", "nemotron3")
    for embedder, model in (("sentence-transformers", "some/model"), ("ollama", "some:model")):
        settings = build_eval_settings(embedder, model, 2048, "http://localhost:11434")
        assert settings.model_profile == "nemotron3", embedder
