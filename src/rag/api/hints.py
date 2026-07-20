"""Map a failed run's log tail to ONE actionable hint — the "what do I do now" line the
UI shows first (the raw log is still there for the details). Pure table lookup, so it
lives apart from the job runner and is trivially testable.
"""
from __future__ import annotations

# Common failure signatures → the first thing to try. Order matters: first match wins.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("out of memory", "mps backend out of memory", "cuda out of memory"),
     "메모리 부족입니다 — batch size를 줄이거나 LoRA로 전환해 보세요. batch를 유지하고 싶으면 Cached MNRL이 "
     "메모리를 아껴줍니다. 데이터에 hard negative가 붙어 있다면 컬럼 수만큼 배치가 무거워지니 max_negatives를 "
     "줄이거나 0으로 끄세요(in-batch negative만 사용). Matryoshka를 켰다면 여러 차원의 backward 그래프를 동시에 "
     "들고 있어 batch·LoRA로도 잘 안 줄어드니, 차원 수를 줄이세요(또는 VRAM이 더 큰 GPU)."),
    (("no module named", "modulenotfounderror"),
     "학습 스택이 설치되어 있지 않습니다 — `uv sync --group training` 후 다시 시도하세요."),
    (("hard negative",),
     "Triplet loss에는 hard negative가 필요합니다 — 데이터 탭에서 hard-negative mining을 켜고 재생성하세요."),
    (("connection refused", "connecterror", "11434"),
     "Ollama가 꺼져 있는 것 같습니다 — `ollama serve`가 실행 중인지 확인하세요."),
    (("no space left",),
     "디스크가 가득 찼습니다 — 모델 페이지에서 안 쓰는 모델을 정리하세요 (런당 약 1GB)."),
)


def hint_for(tail: str) -> str | None:
    """The first hint whose signature appears in the (lower-cased) log tail, or None."""
    lowered = tail.lower()
    for needles, hint in _HINTS:
        if any(needle in lowered for needle in needles):
            return hint
    return None
