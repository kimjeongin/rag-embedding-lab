"""Visual theme + CSS for the web UI (passed to .launch() by cli/webui.py).

A Soft indigo theme + the Pretendard Korean web font, and a small **design system**:
a hero header, card-wrapped sections, a status bar, and KPI cards. Coloured chips
(banners / status pills / KPI deltas) are styled INLINE in `actions.py` with
`!important` so the theme — especially dark mode — can never recolour their text into
invisibility; this stylesheet only handles layout + neutral, theme-variable colours
(which adapt to light/dark automatically).
"""
from __future__ import annotations

import gradio as gr

_FONT_URL = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"

CSS = f"""
@import url('{_FONT_URL}');

.gradio-container {{
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  max-width: 100% !important;
  padding: 4px 26px 48px !important;
}}
footer {{ display: none !important; }}

/* hero header */
.app-hero {{ margin: 8px 2px 12px; }}
.app-hero .t {{ font-size: 25px; font-weight: 800; letter-spacing: -.01em; line-height: 1.2; }}
.app-hero .s {{ color: var(--body-text-color-subdued); font-size: 14px; margin-top: 4px; }}
.app-hero .step {{ color: #4f46e5; font-weight: 700; }}

/* section cards — the main structural unit */
.card {{
  background: var(--block-background-fill) !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
  margin-bottom: 14px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}}
.card h3 {{ margin-top: 2px !important; font-size: 16px !important; }}

/* tab nav */
.tab-nav button {{ font-weight: 600 !important; font-size: 14.5px !important; }}

/* status bar (pills are coloured inline in actions.py) */
.statusbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}

/* KPI cards */
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 2px; }}
.kpi {{
  flex: 1; min-width: 150px; padding: 14px 16px; border-radius: 14px;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
}}
.kpi .label {{ font-size: 12.5px; color: var(--body-text-color-subdued) !important; margin-bottom: 6px; }}
.kpi .value {{ font-size: 26px; font-weight: 800; line-height: 1.1; color: var(--body-text-color) !important; }}

.caption {{ font-size: 12px; color: var(--body-text-color-subdued) !important; margin: 0 0 6px; }}
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
