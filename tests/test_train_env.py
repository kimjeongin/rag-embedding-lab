"""The TrainRequest → TRAIN_* env → TrainingConfig.from_env round-trip.

The job runner can't pass a Python object to its training subprocess, so it serializes
the request to TRAIN_* env vars (rag.api.jobs._train_env) which the child decodes
(rag.training.config.TrainingConfig.from_env). Those two ends are coupled by hand; this
test pins them, so a new field that's encoded-but-not-decoded (or vice-versa) fails here
instead of silently using a default in production.
"""
from rag.api.jobs import _encode_env, _train_env
from rag.api.schemas.lab import TrainRequest
from rag.training.config import TrainingConfig


def test_encode_env_matches_from_env_conventions():
    assert _encode_env(None) == ""            # Optional sentinel
    assert _encode_env(True) == "1"
    assert _encode_env(False) == "0"
    assert _encode_env([512, 256, 64]) == "512,256,64"
    assert _encode_env(2e-5) == "2e-05"


def test_train_request_round_trips_through_env_to_training_config(monkeypatch):
    # deliberately all-non-default so a dropped field can't pass by coincidence
    req = TrainRequest(
        base_model="my/model", output_dir="outputs/x", epochs=7, batch_size=8,
        learning_rate=3e-5, device="cpu", loss="gist", matryoshka=True,
        matryoshka_dims=[512, 128], dropout=0.2, max_negatives=3, early_stop_patience=5,
        early_stop_metric="loss", auto_name=False, seed=7, note="hi",
        method="lora", lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_target="attention",
    )
    for key, value in _train_env(req).items():
        if key.startswith("TRAIN_"):
            monkeypatch.setenv(key, value)

    cfg = TrainingConfig.from_env()

    assert cfg.base_model == "my/model"
    assert cfg.output_dir == "outputs/x"
    assert cfg.epochs == 7
    assert cfg.batch_size == 8
    assert cfg.learning_rate == 3e-5
    assert cfg.device == "cpu"
    assert cfg.loss == "gist"
    assert cfg.matryoshka is True
    assert cfg.matryoshka_dims == (512, 128)
    assert cfg.dropout == 0.2
    assert cfg.max_negatives == 3
    assert cfg.early_stop_patience == 5
    assert cfg.early_stop_metric == "loss"
    assert cfg.auto_name is False
    assert cfg.seed == 7
    assert cfg.note == "hi"
    assert cfg.method == "lora"
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.lora_target == "attention"


def test_optional_dropout_round_trips_as_none(monkeypatch):
    req = TrainRequest(base_model="m", output_dir="o", dropout=None)
    for key, value in _train_env(req).items():
        if key.startswith("TRAIN_"):
            monkeypatch.setenv(key, value)
    cfg = TrainingConfig.from_env()
    assert cfg.dropout is None
    assert cfg.max_negatives is None            # 기본값: 전부 사용


def test_max_negatives_zero_round_trips_as_zero(monkeypatch):
    """0은 '제외(in-batch만)'라는 실값 — Optional의 "" 센티널과 섞이면 안 된다."""
    req = TrainRequest(base_model="m", output_dir="o", max_negatives=0)
    for key, value in _train_env(req).items():
        if key.startswith("TRAIN_"):
            monkeypatch.setenv(key, value)
    assert TrainingConfig.from_env().max_negatives == 0
