"""Output naming + env parsing for training — no torch needed (heavy imports live
inside train(), so rag.training.train imports clean)."""
from rag.training.config import TrainingConfig
from rag.training.train import _compose_output_dir


def test_auto_name_embeds_loss_and_saved_epoch():
    cfg = TrainingConfig(output_dir="outputs/embedding-ft", loss="mnrl")
    assert _compose_output_dir(cfg, 7) == "outputs/embedding-ft-mnrl-e7"


def test_auto_name_adds_rank_for_lora():
    cfg = TrainingConfig(output_dir="outputs/embedding-ft-lora", loss="gist", method="lora", lora_r=8)
    assert _compose_output_dir(cfg, 5) == "outputs/embedding-ft-lora-gist-r8-e5"


def test_auto_name_never_overwrites_an_existing_model(tmp_path):
    cfg = TrainingConfig(output_dir=str(tmp_path / "ft"), loss="mnrl")
    (tmp_path / "ft-mnrl-e3").mkdir()
    (tmp_path / "ft-mnrl-e3-2").mkdir()
    assert _compose_output_dir(cfg, 3) == str(tmp_path / "ft-mnrl-e3-3")


def test_auto_name_off_returns_output_dir_verbatim():
    cfg = TrainingConfig(output_dir="outputs/exact-name", auto_name=False)
    assert _compose_output_dir(cfg, 7) == "outputs/exact-name"


def test_from_env_reads_the_new_knobs(monkeypatch):
    for key, value in {
        "TRAIN_LOSS": "triplet",
        "TRAIN_DROPOUT": "0.15",
        "TRAIN_PATIENCE": "5",
        "TRAIN_MONITOR": "loss",
        "TRAIN_AUTO_NAME": "0",
        "TRAIN_LORA_TARGET": "attention",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = TrainingConfig.from_env()
    assert cfg.loss == "triplet"
    assert cfg.dropout == 0.15
    assert cfg.early_stop_patience == 5
    assert cfg.early_stop_metric == "loss"
    assert cfg.auto_name is False
    assert cfg.lora_target == "attention"


def test_from_env_blank_dropout_keeps_model_defaults(monkeypatch):
    monkeypatch.setenv("TRAIN_DROPOUT", "")
    assert TrainingConfig.from_env().dropout is None
