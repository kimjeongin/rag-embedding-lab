"""Visual theme + CSS for the web UI (passed to .launch() by cli/webui.py).

A Soft indigo theme + the Pretendard Korean web font. The layout is deliberately FLAT
(no boxed "cards" / heavy borders); structure comes from headings + whitespace. Coloured
chips (banners / status pills / KPI deltas) are styled INLINE with `!important` in
`actions.py` so the theme — especially dark mode — can never recolour their text into
invisibility. The full-data popup is a fixed overlay toggled by a `show` class from JS
(reliable open/close + ESC), not Gradio's `visible` (which left a stuck backdrop).
"""
from __future__ import annotations

import gradio as gr

_FONT_URL = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"

CSS = f"""
@import url('{_FONT_URL}');

.gradio-container {{
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  max-width: 100% !important;
  padding: 4px 24px 44px !important;
}}
footer {{ display: none !important; }}

/* hero header (text only, no box) */
.app-hero {{ margin: 8px 2px 12px; }}
.app-hero .t {{ font-size: 24px; font-weight: 800; letter-spacing: -.01em; line-height: 1.2; }}
.app-hero .s {{ color: var(--body-text-color-subdued); font-size: 13.5px; margin-top: 3px; }}
.app-hero .step {{ color: #4f46e5; font-weight: 700; }}

/* status bar (pills coloured inline in actions.py) */
.statusbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}

/* KPI result cards — soft fill, NO border */
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 2px; }}
.kpi {{ flex: 1; min-width: 150px; padding: 14px 16px; border-radius: 14px; background: var(--background-fill-secondary); }}
.kpi .label {{ font-size: 12.5px; color: var(--body-text-color-subdued) !important; margin-bottom: 6px; }}
.kpi .value {{ font-size: 26px; font-weight: 800; line-height: 1.1; color: var(--body-text-color) !important; }}

.caption {{ font-size: 12px; color: var(--body-text-color-subdued) !important; margin: 0 0 6px; }}

/* full-data popup — hidden by default; JS adds `.show` to display it */
.rag-modal {{
  display: none !important; position: fixed; inset: 0; z-index: 9999;
  background: rgba(15, 23, 42, .5); padding: 4vh 5vw; overflow: auto;
}}
.rag-modal.show {{ display: block !important; }}
.rag-modal .modal-inner {{
  background: var(--background-fill-primary); border: 1px solid rgba(128,128,128,.2);
  border-radius: 16px; padding: 14px 20px 20px; max-width: 1080px; margin: 0 auto;
  box-shadow: 0 20px 60px rgba(0,0,0,.35);
}}
"""


def lab_theme() -> gr.Theme:
    """The app's Gradio theme — Soft indigo, rounded, Pretendard."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Pretendard", "system-ui", "sans-serif"],
        radius_size="lg",
        spacing_size="md",
    )
