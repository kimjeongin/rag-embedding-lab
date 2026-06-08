"""The Gradio Blocks UI — generate data → train → evaluate → compare.

The only module that imports gradio. All data/logic lives in `rag.webui.actions`
(gradio-free); this file is layout + copy + wiring. Layout is flat (headings + spacing,
no boxed cards). The "full data" view is a real popup: a fixed overlay shown/hidden by
JS toggling a `show` class (reliable open/close + ESC/backdrop), since Gradio's `visible`
toggle left a stuck backdrop. Radio/Dropdown use (label, value) pairs so display text is
decoupled from the stable value handlers switch on.
"""
from __future__ import annotations

import gradio as gr
import pandas as pd

from rag.webui import actions

_LOSS_COLS = ["step", "epoch", "loss"]
_HERO = (
    "<div class='app-hero'>"
    "<div class='t'>🧪 RAG Embedding Lab</div>"
    "<div class='s'>검색용 임베딩 모델을 <span class='step'>① 데이터 → ② 학습 → ③ 평가 → ④ 비교</span> "
    "한 곳에서. 각 단계는 앞 단계가 만든 파일을 그대로 이어받습니다.</div>"
    "</div>"
)
_GLOSSARY = (
    "- **recall@k** — 정답 문서가 상위 k개 안에 든 질문 비율 (recall@1 = 1등으로 맞힘)\n"
    "- **MRR@10** — 정답의 평균 역순위 (1에 가까울수록 상위에 위치)\n"
    "- **nDCG@10** — 순위 품질 종합 점수 (0~1)\n\n→ 모두 **높을수록 좋음.**"
)
# pure client-side: open by id, close all, and (once) wire ESC + backdrop-click to close
_OPEN_PAIRS = "() => document.getElementById('modal_pairs').classList.add('show')"
_OPEN_CORPUS = "() => document.getElementById('modal_corpus').classList.add('show')"
_CLOSE_PAIRS = "() => document.getElementById('modal_pairs').classList.remove('show')"
_CLOSE_CORPUS = "() => document.getElementById('modal_corpus').classList.remove('show')"
_MODAL_INIT = """() => {
  if (window.__ragModalInit) return; window.__ragModalInit = true;
  const closeAll = () => document.querySelectorAll('.rag-modal.show').forEach(m => m.classList.remove('show'));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAll(); });
  document.addEventListener('click', e => { if (e.target.classList && e.target.classList.contains('rag-modal')) closeAll(); });
}"""


def _refresh_compare():
    return actions.compare_styled(), actions.compare_figure()


def _on_table_select(evt: gr.SelectData):
    # the leading 🗑 column (index 0) deletes that row; other cells do nothing destructive
    if evt.index and evt.index[1] == 0:
        actions.delete_run_at(evt.index[0])
    return _refresh_compare()


def _models_update(embedder: str, ollama_url: str):
    choices = actions.list_models(embedder, ollama_url)
    return gr.update(choices=choices, value=actions.default_model(embedder, choices))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Embedding Lab", fill_width=True) as demo:
        gr.HTML(_HERO)
        with gr.Row():
            status = gr.HTML(actions.status_html)
            status_btn = gr.Button("⟳", scale=0, min_width=46)

        with gr.Tabs():
            # ── 1) DATA ──────────────────────────────────────────────────────
            with gr.Tab("① 데이터", id="tab_data"):
                gr.Markdown(
                    "여기서 만든 데이터가 다음 단계로 흘러갑니다 — "
                    "**학습 데이터 → ② 학습**, **평가 데이터 → ③ 평가**."
                )
                gr.Markdown("### 📦 현재 보유 데이터")
                gr.Markdown(
                    "‘개수’ = 각 파일의 레코드 수 (학습쌍 = 짝 수 · corpus = 문서 수 · queries = 질문 수).",
                    elem_classes="caption",
                )
                d_overview = gr.Dataframe(actions.data_overview, interactive=False)

                gr.Markdown("### 📚 학습 데이터　→ ② 학습에서 사용")
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

                gr.Markdown("### 🧪 평가 데이터　→ ③ 평가에서 사용")
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
                    "base 모델을 **① 데이터**의 학습 데이터로 **fine-tuning** 합니다. "
                    "아래 **loss가 내려가면** 정상이에요."
                )
                t_ready = gr.HTML(actions.training_status_html)
                with gr.Group(visible=not actions.training_ready()) as t_install_grp:
                    gr.Markdown(
                        "학습에는 추가 라이브러리(torch 등)가 필요해요. 터미널 없이 "
                        "**아래 버튼으로 한 번만** 설치하면 됩니다."
                    )
                    t_install_btn = gr.Button("📦 학습 라이브러리 설치 (한 번만)", variant="primary")
                    t_install_log = gr.Textbox(label="설치 로그", lines=8, visible=False)

                gr.Markdown("### ⚙️ 학습 설정")
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

                gr.Markdown("### 📈 진행 상황")
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
                    e_refresh_models = gr.Button("⟳", scale=0, min_width=46)
                e_label = gr.Textbox(
                    "", label="run 라벨",
                    info="비교 표에서 구분할 이름(비우면 모델명). embedding 차원은 자동 감지돼요.",
                )
                with gr.Accordion("고급 설정", open=False):
                    e_ollama = gr.Textbox("http://localhost:11434", label="Ollama URL")
                    e_dir = gr.Textbox("data/eval", label="평가셋 폴더 (EVAL_DIR)")
                e_run = gr.Button("평가 실행", variant="primary")
                gr.Markdown("### 📊 결과")
                e_status = gr.Markdown()
                e_kpi = gr.HTML()

            # ── 4) COMPARE ───────────────────────────────────────────────────
            with gr.Tab("④ 비교", id="tab_compare"):
                gr.Markdown(
                    "**③ 평가**들을 한눈에 비교합니다. 표는 **지표별 1등을 초록색**으로, 그래프는 "
                    "**모든 지표를 모델별 막대로 나란히**. 표의 **🗑 칸을 클릭**하면 그 기록이 삭제돼요."
                )
                with gr.Accordion("📖 지표 설명", open=False):
                    gr.Markdown(_GLOSSARY)
                c_refresh = gr.Button("⟳ 새로고침", scale=0)
                c_table = gr.Dataframe(
                    actions.compare_styled, label="평가 결과 (🗑 클릭 = 삭제 · 초록 = 지표별 1등)", interactive=False,
                )
                c_plot = gr.Plot(actions.compare_figure)

        # ── full-data popups (fixed overlays toggled by JS) ───────────────────
        with gr.Group(elem_classes="rag-modal", elem_id="modal_pairs"):
            with gr.Column(elem_classes="modal-inner"):
                with gr.Row():
                    gr.Markdown("### 학습 데이터 — 전체")
                    pairs_close = gr.Button("✕ 닫기", scale=0, min_width=80)
                modal_pairs_df = gr.Dataframe(actions.full_pairs, interactive=False)
        with gr.Group(elem_classes="rag-modal", elem_id="modal_corpus"):
            with gr.Column(elem_classes="modal-inner"):
                with gr.Row():
                    gr.Markdown("### 평가 corpus — 전체")
                    corpus_close = gr.Button("✕ 닫기", scale=0, min_width=80)
                modal_corpus_df = gr.Dataframe(actions.full_corpus, interactive=False)

        # ── wiring ───────────────────────────────────────────────────────────
        demo.load(None, None, None, js=_MODAL_INIT)
        status_btn.click(actions.status_html, None, status)

        # data
        d_method.change(lambda m: gr.update(visible=m == "synthetic"), d_method, d_syn)
        d_pairs_btn.click(
            actions.gen_pairs, [d_method, d_corpus, d_genmodel, d_nq, d_hn], [d_pairs_status, d_pairs_prev]
        ).then(actions.full_pairs, None, modal_pairs_df).then(
            actions.data_overview, None, d_overview
        ).then(actions.train_data_info, None, t_datainfo).then(actions.status_html, None, status)
        d_eval_btn.click(actions.gen_eval_set, [d_ndist], [d_eval_status, d_eval_prev]).then(
            actions.full_corpus, None, modal_corpus_df
        ).then(actions.data_overview, None, d_overview).then(
            actions.eval_header_html, None, e_banner
        ).then(actions.status_html, None, status)
        d_pairs_detail.click(None, None, None, js=_OPEN_PAIRS)
        pairs_close.click(None, None, None, js=_CLOSE_PAIRS)
        d_eval_detail.click(None, None, None, js=_OPEN_CORPUS)
        corpus_close.click(None, None, None, js=_CLOSE_CORPUS)

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
            actions.run_eval, [e_embedder, e_model, e_ollama, e_dir, e_label], [e_status, e_kpi]
        ).then(_refresh_compare, None, [c_table, c_plot]).then(actions.status_html, None, status)

        # compare
        c_refresh.click(_refresh_compare, None, [c_table, c_plot])
        c_table.select(_on_table_select, None, [c_table, c_plot])

    return demo
