"""Web UI (Gradio) — a delivery layer over the offline pipeline.

Like `cli/` and `api/`, this package only *orchestrates* the existing rag.* functions
(datagen / training / evaluation) — no business logic lives here. It wraps them in a
"generate data → train → evaluate → compare" UI for people who'd rather not drive the
CLI.

Import layout (so the package stays testable without gradio installed):
  - `runs`    — the eval-run registry (stdlib only)
  - `jobs`    — stream a child command's output (stdlib only)
  - `actions` — glue to rag.* (pandas + rag stack, NO gradio)
  - `app`     — the Gradio Blocks (the only module that imports gradio)

This `__init__` intentionally imports nothing, so `import rag.webui.runs` works in the
test environment (which doesn't install the `ui` group). Launch with `rag-ui`.
"""
