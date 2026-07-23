"""Which input format does THIS model want — the one place that decides.

`core.formatting` defines the templates but stays pure; picking a profile needs env
and disk, so it lives here. One resolver, used by training, evaluation and serving
alike, because a model formatted one way at train time and another way at serve time
fails silently (see `core.formatting`).

The subtle case is a **fine-tuned model**: it's a filesystem PATH
(``outputs/embedding-ft-mnrl-e8``), not a model name, so name matching can't see what
architecture it came from. Training records ``base_model`` in ``train_meta.json``, so
a tuned model inherits its base's profile — the same trick `lab.infer_dim` uses to
read the dimension off ``1_Pooling/config.json`` instead of loading the model.

Resolution order (first hit wins):

  1. explicit override (``MODEL_PROFILE`` / ``TRAIN_MODEL_PROFILE``) — always obeyed
  2. the model dir's ``train_meta.json``: its recorded ``model_profile``, else the
     profile implied by its ``base_model``
  3. the model name itself ("nvidia/Nemotron-3-Embed-1B-BF16" → nemotron3)
  4. the historical default (qwen3), announced on stderr — never silent
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.core.formatting import (
    DEFAULT_PROFILE,
    PROFILES,
    ModelProfile,
    profile_for_name,
)


class UnknownProfile(ValueError):
    """An explicit profile override that isn't one of the defined profiles."""


def _from_meta(model: str) -> ModelProfile | None:
    """The profile recorded next to a fine-tuned model, or implied by its base."""
    meta_file = Path(model) / "train_meta.json"
    if not meta_file.is_file():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None                              # unreadable meta: fall through
    recorded = meta.get("model_profile")
    if recorded in PROFILES:
        return PROFILES[recorded]
    base = meta.get("base_model")
    return profile_for_name(str(base)) if base else None


def resolve_profile(model: str, override: str = "") -> ModelProfile:
    """The `ModelProfile` for ``model``; ``override`` is a profile name that wins."""
    if override:
        if override not in PROFILES:
            raise UnknownProfile(
                f"알 수 없는 모델 프로파일 '{override}' — {sorted(PROFILES)} 중에서 고르세요"
            )
        return PROFILES[override]

    resolved = _from_meta(model) or profile_for_name(model)
    if resolved is not None:
        return resolved

    print(
        f"[profile] '{model}'이 알려진 모델 패턴과 맞지 않아 기본 프로파일"
        f"({DEFAULT_PROFILE.name})을 사용합니다 — 입력 포맷이 다르면 점수가 조용히 "
        f"나빠지니 MODEL_PROFILE로 명시하세요 (선택지: {sorted(PROFILES)})"
    )
    return DEFAULT_PROFILE
