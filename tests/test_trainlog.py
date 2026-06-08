"""Training-log parsing — turns rag-train stdout into points + before/after nDCG."""
from rag.trainlog import clean_tqdm, parse_eval_ndcg, parse_loss_points


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
