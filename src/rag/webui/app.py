"""The Gradio Blocks UI — generate data → train → evaluate → compare.

The only module that imports gradio. All data/logic lives in `rag.webui.actions`
(gradio-free); this file is layout + copy + wiring. Each tab shows the concrete data it
consumes (file + counts) so the data → train → eval → compare flow is visible, not just
implied by tab order. Radio/Dropdown use (label, value) pairs so display text is
decoupled from the stable value handlers switch on.
"""
from __future__ import annotations

import gradio as gr
import pandas as pd

from rag.webui import actions

_INTRO = (
    "# 🧪 RAG Embedding Lab\n"
    "검색용 임베딩 모델을 학습·평가하는 실험실. **① 데이터 → ② 학습 → ③ 평가 → ④ 비교** 순서로 진행하세요. "
    "각 단계는 앞 단계가 만든 파일을 그대로 이어받습니다."
)
_LOSS_COLS = ["step", "epoch", "loss"]
_GLOSSARY = (
    "- **recall@k** — 정답 문서가 상위 k개 안에 든 질문 비율 (recall@1 = 1등으로 맞힘)\n"
    "- **MRR@10** — 정답의 평균 역순위 (1에 가까울수록 상위에 위치)\n"
    "- **nDCG@10** — 순위 품질 종합 점수 (0~1)\n\n"
    "→ 모두 **높을수록 좋음.**"
)


def _all_plot() -> gr.BarPlot:
    df, y_lim = actions.compare_all_data()
    return gr.BarPlot(
        df, x="metric", y="value", color="run", y_lim=y_lim,
        title="지표별 점수 (모델별 색 · 높을수록 좋음)", height=340,
    )


def _refresh_compare():
    return actions.compare_styled(), _all_plot()


def _on_row_select(evt: gr.SelectData):
    return evt.index[0] if evt.index is not None else None


def _delete_selected(row_index):
    actions.delete_run_at(row_index)
    table, plot = _refresh_compare()
    return table, plot, None


def _models_update(embedder: str, ollama_url: str):
    return gr.update(choices=actions.list_models(embedder, ollama_url))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Embedding Lab", fill_width=True) as demo:
        gr.Markdown(_INTRO)
        with gr.Row():
            status = gr.HTML(actions.status_html)
            status_btn = gr.Button("⟳", scale=0, min_width=50)

        with gr.Tabs():
            # ── 1) DATA ──────────────────────────────────────────────────────
            with gr.Tab("① 데이터", id="tab_data"):
                gr.Markdown(
                    "여기서 만든 데이터가 다음 단계로 흘러갑니다 — "
                    "**학습 데이터 → ② 학습**, **평가 데이터 → ③ 평가**."
                )
                with gr.Accordion("📦 현재 보유 데이터 — 각 데이터가 어느 단계에 쓰이는지", open=True):
                    gr.Markdown(
                        "‘개수’는 각 파일에 든 **레코드 수**예요 "
                        "(학습쌍 = (질문, 정답) 짝 수 · corpus = 문서 수 · queries = 질문 수).",
                        elem_classes="caption",
                    )
                    d_overview = gr.Dataframe(actions.data_overview, interactive=False)

                gr.Markdown("### 1. 학습 데이터　→ ② 학습에서 사용")
                d_method = gr.Radio(
                    choices=[("예제 데이터 (미리 만든 샘플)", "toy"), ("AI 자동 생성 (내 문서 기반)", "synthetic")],
                    value="toy", label="생성 방식",
                    info="예제 = 도구를 빠르게 테스트할 미리 만든 데이터 · "
                    "AI 자동 생성 = 내 문서(corpus)를 LLM이 읽고 질문을 작성",
                )
                with gr.Group(visible=False) as d_syn:
                    with gr.Row():
                        d_corpus = gr.Textbox("data/corpus.jsonl", label="corpus 파일")
                        d_genmodel = gr.Textbox("qwen3:4b", label="질문 생성 LLM")
                    with gr.Row():
                        d_nq = gr.Number(3, label="문서당 질문 수", precision=0)
                        d_hn = gr.Number(1, label="hard negative 수", precision=0, info="정답과 헷갈리는 오답")
                d_pairs_btn = gr.Button("학습 데이터 생성", variant="primary")
                d_pairs_status = gr.Markdown()
                d_pairs_prev = gr.Dataframe(label="미리보기 (앞 8개)", interactive=False)
                d_pairs_detail = gr.Button("📋 전체 보기", size="sm")

                gr.Markdown("### 2. 평가 데이터　→ ③ 평가에서 사용")
                d_ndist = gr.Slider(
                    16, 448, value=448, step=16, label="distractor 수",
                    info="정답 외 ‘방해 문서’. 많을수록 난이도가 올라가 모델 차이가 잘 드러납니다.",
                )
                d_eval_btn = gr.Button("평가 데이터 생성", variant="primary")
                d_eval_status = gr.Markdown()
                d_eval_prev = gr.Dataframe(label="corpus 미리보기 (앞 8개)", interactive=False)
                d_eval_detail = gr.Button("📋 전체 보기", size="sm")

            # ── 2) TRAIN ─────────────────────────────────────────────────────
            with gr.Tab("② 학습", id="tab_train"):
                gr.Markdown(
                    "base 모델을 **① 데이터**의 학습 데이터로 **fine-tuning** 합니다. 아래 **loss가 내려가면** 정상이에요."
                )
                t_ready = gr.HTML(actions.training_status_html)
                with gr.Group(visible=not actions.training_ready()) as t_install_grp:
                    gr.Markdown(
                        "학습에는 추가 라이브러리(torch 등)가 필요해요. 터미널 없이 **아래 버튼으로 한 번만** 설치하면 됩니다."
                    )
                    t_install_btn = gr.Button("📦 학습 라이브러리 설치 (한 번만)", variant="primary")
                    t_install_log = gr.Textbox(label="설치 로그", lines=8, visible=False)
                t_datainfo = gr.HTML(actions.train_data_info)
                with gr.Row():
                    t_base = gr.Textbox("Qwen/Qwen3-Embedding-0.6B", label="base 모델 (HuggingFace)")
                    t_out = gr.Textbox("outputs/embedding-ft", label="저장 폴더")
                with gr.Row():
                    t_epochs = gr.Number(1, label="epochs", precision=0)
                    t_batch = gr.Number(16, label="batch size", precision=0)
                    t_lr = gr.Number(2e-5, label="learning rate")
                    t_device = gr.Dropdown(["", "mps", "cuda", "cpu"], value="", label="device (빈칸=auto)")
                t_run = gr.Button("학습 시작", variant="primary")
                t_kpi = gr.HTML()
                t_loss = gr.LinePlot(
                    value=pd.DataFrame(columns=_LOSS_COLS), x="step", y="loss",
                    title="training loss (낮을수록 좋음)", height=280,
                )
                t_log = gr.Textbox(label="train log", lines=16, max_lines=16, autoscroll=True)

            # ── 3) EVALUATE ──────────────────────────────────────────────────
            with gr.Tab("③ 평가", id="tab_eval"):
                gr.Markdown(
                    "**② 학습**의 모델(또는 기본 모델)이 **① 데이터**의 평가셋에서 정답 문서를 얼마나 잘 검색하는지 "
                    "측정합니다. 결과는 **④ 비교**에 누적돼요."
                )
                e_banner = gr.HTML(actions.eval_header_html)
                with gr.Row():
                    e_embedder = gr.Radio(
                        choices=[
                            ("Ollama (기본 모델)", "ollama"),
                            ("sentence-transformers (학습한 모델)", "sentence-transformers"),
                        ],
                        value="ollama", label="백엔드",
                    )
                    e_model = gr.Dropdown(
                        choices=actions.list_models("ollama", "http://localhost:11434"),
                        value="qwen3-embedding:0.6b", label="모델", allow_custom_value=True, scale=2,
                    )
                    e_refresh_models = gr.Button("⟳", scale=0, min_width=50)
                e_label = gr.Textbox(
                    "", label="run 라벨", info="비교 표에서 구분할 이름(비우면 모델명). embedding 차원은 자동 감지돼요.",
                )
                with gr.Accordion("고급", open=False):
                    e_ollama = gr.Textbox("http://localhost:11434", label="Ollama URL")
                    e_dir = gr.Textbox("data/eval", label="평가셋 폴더 (EVAL_DIR)")
                e_run = gr.Button("평가 실행", variant="primary")
                e_status = gr.Markdown()
                e_kpi = gr.HTML()

            # ── 4) COMPARE ───────────────────────────────────────────────────
            with gr.Tab("④ 비교", id="tab_compare"):
                gr.Markdown(
                    "**③ 평가**들을 한눈에 비교합니다. 표는 **지표별 1등을 초록색**으로, 그래프는 **모든 지표를 한 번에** "
                    "(축을 데이터에 맞춰 확대해 작은 차이도 보이게)."
                )
                with gr.Accordion("📖 지표 설명", open=False):
                    gr.Markdown(_GLOSSARY)
                c_refresh = gr.Button("⟳ 새로고침", scale=0)
                c_table = gr.Dataframe(
                    actions.compare_styled,
                    label="평가 결과 (초록 = 지표별 1등 · 삭제하려면 행을 클릭)",
                    interactive=False,
                )
                c_plot = _all_plot()
                with gr.Row():
                    c_del_btn = gr.Button("🗑 선택한 행 삭제", scale=0)
                    gr.Markdown("표에서 삭제할 run의 **행을 클릭**한 뒤 버튼을 누르세요.")
                c_selected = gr.State(None)

        # ── 'view all' modals (fixed overlays toggled visible) ────────────────
        with gr.Group(visible=False, elem_classes="modal") as modal_pairs:
            with gr.Column(elem_classes="modal-card"):
                gr.Markdown("### 학습 데이터 — 전체")
                modal_pairs_close = gr.Button("✕ 닫기", scale=0)
                modal_pairs_df = gr.Dataframe(interactive=False)
        with gr.Group(visible=False, elem_classes="modal") as modal_corpus:
            with gr.Column(elem_classes="modal-card"):
                gr.Markdown("### 평가 corpus — 전체")
                modal_corpus_close = gr.Button("✕ 닫기", scale=0)
                modal_corpus_df = gr.Dataframe(interactive=False)

        # ── wiring ───────────────────────────────────────────────────────────
        status_btn.click(actions.status_html, None, status)

        # data
        d_method.change(lambda m: gr.update(visible=m == "synthetic"), d_method, d_syn)
        d_pairs_btn.click(
            actions.gen_pairs, [d_method, d_corpus, d_genmodel, d_nq, d_hn], [d_pairs_status, d_pairs_prev]
        ).then(actions.data_overview, None, d_overview).then(
            actions.train_data_info, None, t_datainfo
        ).then(actions.status_html, None, status)
        d_eval_btn.click(actions.gen_eval_set, [d_ndist], [d_eval_status, d_eval_prev]).then(
            actions.data_overview, None, d_overview
        ).then(actions.eval_header_html, None, e_banner).then(actions.status_html, None, status)

        d_pairs_detail.click(lambda: (gr.update(visible=True), actions.full_pairs()), None, [modal_pairs, modal_pairs_df])
        modal_pairs_close.click(lambda: gr.update(visible=False), None, modal_pairs)
        d_eval_detail.click(lambda: (gr.update(visible=True), actions.full_corpus()), None, [modal_corpus, modal_corpus_df])
        modal_corpus_close.click(lambda: gr.update(visible=False), None, modal_corpus)

        # train
        t_install_btn.click(lambda: gr.update(visible=True), None, t_install_log).then(
            actions.install_training, None, t_install_log
        ).then(actions.training_status_html, None, t_ready).then(
            lambda: gr.update(visible=not actions.training_ready()), None, t_install_grp
        )
        t_run.click(actions.run_train, [t_base, t_epochs, t_batch, t_lr, t_out, t_device], [t_log, t_loss, t_kpi])

        # eval
        e_embedder.change(_models_update, [e_embedder, e_ollama], e_model)
        e_refresh_models.click(_models_update, [e_embedder, e_ollama], e_model)
        e_run.click(
            actions.run_eval,
            [e_embedder, e_model, e_ollama, e_dir, e_label],
            [e_status, e_kpi],
        ).then(_refresh_compare, None, [c_table, c_plot]).then(actions.status_html, None, status)

        # compare
        c_refresh.click(_refresh_compare, None, [c_table, c_plot])
        c_table.select(_on_row_select, None, c_selected)
        c_del_btn.click(_delete_selected, c_selected, [c_table, c_plot, c_selected])

    return demo
