"""`rag-ui` — launch the Gradio web UI (rag.webui).

Needs the `ui` dependency group (`uv sync --group ui`); the Train tab additionally
needs `--group training`. Host/port via UI_HOST / UI_PORT.
"""
from __future__ import annotations

import os


def main() -> None:
    from rag.webui.app import build_ui
    from rag.webui.theme import CSS, lab_theme

    build_ui().queue().launch(
        server_name=os.getenv("UI_HOST", "127.0.0.1"),
        server_port=int(os.getenv("UI_PORT", "7860")),
        theme=lab_theme(),
        css=CSS,
        show_error=True,
    )
