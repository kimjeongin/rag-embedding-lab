"""Visual theme + CSS for the web UI (passed to .launch() by cli/webui.py).

A Soft indigo theme + the Pretendard Korean web font, plus a CSS layer for the status
bar, KPI cards, and banners.

Every coloured chip/banner sets BOTH its background and text colour with `!important`,
because Gradio's theme (especially dark mode) otherwise overrides the text colour to a
light value — leaving light text on a light chip (invisible). Nested <code>/<span> are
pinned too, so file paths inside banners stay readable.
"""
from __future__ import annotations

import gradio as gr

# Pretendard (Korean-friendly) from a CDN; falls back to system sans if offline.
_FONT_URL = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"

CSS = f"""
@import url('{_FONT_URL}');

/* full-width layout with comfortable side gutters */
.gradio-container {{
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  max-width: 100% !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
}}
footer {{ display: none !important; }}

/* status bar — light chip + dark text, forced so the theme can't recolour it */
.statusbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:2px 0 10px; }}
.statusbar .pill {{
  font-size:12.5px; padding:5px 12px; border-radius:999px; white-space:nowrap;
  background:#eef2ff !important; border:1px solid #c7d2fe !important; color:#3730a3 !important;
}}
.statusbar .pill.ok   {{ background:#e7f6ec !important; border-color:#a7d8b8 !important; color:#14532d !important; }}
.statusbar .pill.warn {{ background:#fff4e5 !important; border-color:#ffce85 !important; color:#7a4a00 !important; }}

/* banners */
.banner {{ padding:11px 15px; border-radius:12px; font-size:13.5px; line-height:1.55; margin:6px 0; }}
.banner.warn {{ background:#fff4e5 !important; border:1px solid #ffce85 !important; color:#7a4a00 !important; }}
.banner.info {{ background:#eef2ff !important; border:1px solid #c7d2fe !important; color:#3730a3 !important; }}
.banner b {{ font-weight:700; }}

/* keep inline <code>/<span> readable INSIDE coloured banners + pills (inherit the
   banner's dark text instead of the theme's code/text colour) */
.banner code, .statusbar code {{
  background:rgba(0,0,0,.07) !important; color:inherit !important;
  padding:1px 6px; border-radius:5px; font-size:.92em;
}}
.banner span, .banner a, .statusbar span {{ color:inherit !important; }}

/* KPI cards (these use theme vars, which already adapt to light/dark) */
.kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin:8px 0; }}
.kpi {{
  flex:1; min-width:160px; padding:14px 16px; border-radius:14px;
  background:var(--block-background-fill); border:1px solid var(--border-color-primary);
  box-shadow:0 1px 3px rgba(0,0,0,.06);
}}
.kpi .label {{ font-size:12.5px; color:var(--body-text-color-subdued) !important; margin-bottom:6px; }}
.kpi .value {{ font-size:27px; font-weight:700; line-height:1.1; color:var(--body-text-color) !important; }}
.kpi .delta {{
  display:inline-block; margin-top:7px; font-size:12.5px; font-weight:700;
  padding:2px 9px; border-radius:999px;
}}
.kpi .delta.up   {{ background:#e7f6ec !important; color:#14532d !important; }}
.kpi .delta.down {{ background:#fdecec !important; color:#8a1c1c !important; }}
.kpi .delta.flat {{ background:#eef1f5 !important; color:#475569 !important; }}

/* 'view all' modal — a fixed overlay we toggle visible */
.modal {{
  position:fixed !important; inset:0 !important; z-index:1000 !important;
  background:rgba(15,23,42,.55) !important; border:none !important;
  padding:4vh 6vw !important; overflow:auto !important;
}}
.modal-card {{
  background:var(--background-fill-primary); border:1px solid var(--border-color-primary);
  border-radius:16px; padding:18px 20px; max-width:1150px; margin:0 auto;
  box-shadow:0 24px 64px rgba(0,0,0,.4);
}}
.caption {{ font-size:12px; color:var(--body-text-color-subdued) !important; margin:0 0 6px; }}
"""


def lab_theme() -> gr.Theme:
    """The app's Gradio theme."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Pretendard", "system-ui", "sans-serif"],
    )
