"""Training-log parsing — turns rag-train stdout into points + before/after nDCG."""
from rag.trainlog import (
    clean_tqdm,
    parse_epoch_points,
    parse_eval_ndcg,
    parse_loss_points,
    parse_saved_path,
    parse_summary,
)


def test_clean_tqdm_keeps_only_the_final_redraw_per_line():
    assert clean_tqdm("a\rb\rc\nd") == "c\nd"
    assert clean_tqdm("plain") == "plain"


def test_parse_loss_points_reads_quoted_and_unquoted():
    text = "\n".join(
        [
            "noise",
            "{'loss': 0.5, 'epoch': 1.0, 'step': 1}",   # unquoted
            "{'loss': '0.30', 'epoch': '2.0'}",          # transformers v5 quoted
        ]
    )
    points = parse_loss_points(text)
    assert [p["step"] for p in points] == [1, 2]
    assert [p["loss"] for p in points] == [0.5, 0.30]
    assert [p["epoch"] for p in points] == [1.0, 2.0]


def test_parse_loss_points_falls_back_to_step_when_no_epoch():
    points = parse_loss_points("{'loss': 0.9}")
    assert points == [{"step": 1, "epoch": 1.0, "loss": 0.9}]


def test_parse_eval_ndcg_picks_baseline_then_after():
    text = "\n".join(
        [
            "running baseline eval",
            "recall@1 = 0.70  ndcg@10 = 0.80",
            "... fine-tuning ...",
            "after fine-tuning",
            "ndcg@10 = 0.92",
        ]
    )
    assert parse_eval_ndcg(text) == (0.80, 0.92)


def test_parse_eval_ndcg_none_when_absent():
    assert parse_eval_ndcg("no metrics here") == (None, None)


def test_parse_epoch_points_reads_per_epoch_validation_lines():
    text = "\n".join(
        [
            "noise",
            "[epoch] 1/12 eval_loss=0.5000 ndcg@10=0.9000 best=1",
            "{'loss': 0.4, 'epoch': 1.5}",
            "[epoch] 2/12 eval_loss=0.4100 ndcg@10=0.9300 best=2",
        ]
    )
    points = parse_epoch_points(text)
    assert points == [
        {"epoch": 1, "max_epochs": 12, "eval_loss": 0.5, "ndcg": 0.9, "best_epoch": 1},
        {"epoch": 2, "max_epochs": 12, "eval_loss": 0.41, "ndcg": 0.93, "best_epoch": 2},
    ]


def test_parse_epoch_points_treats_dash_as_missing_metric():
    points = parse_epoch_points("[epoch] 3/8 eval_loss=- ndcg@10=0.9100 best=2")
    assert points == [{"epoch": 3, "max_epochs": 8, "eval_loss": None, "ndcg": 0.91, "best_epoch": 2}]


def test_parse_saved_path_returns_the_final_auto_named_dir():
    text = "\n".join(
        [
            "[train] summary best_epoch=7 ran=10 early_stopped=yes",
            "[train] saved fine-tuned model to outputs/embedding-ft-mnrl-e7",
        ]
    )
    assert parse_saved_path(text) == "outputs/embedding-ft-mnrl-e7"
    assert parse_saved_path("nothing saved") is None


def test_parse_summary_reads_best_ran_and_early_stop():
    text = "[train] summary best_epoch=7 ran=10 early_stopped=yes"
    assert parse_summary(text) == {"best_epoch": 7, "ran": 10, "early_stopped": True}
    assert parse_summary("[train] summary best_epoch=12 ran=12 early_stopped=no") == {
        "best_epoch": 12,
        "ran": 12,
        "early_stopped": False,
    }
    assert parse_summary("no summary") is None
