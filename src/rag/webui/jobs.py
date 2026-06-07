"""Run a child command and stream its combined output, line by line.

Used by the UI's Train tab: training is long and writes progress to stdout/stderr, so
we launch it as a subprocess — which also keeps torch out of the UI process — and yield
the accumulated log for Gradio to render live. Stdlib only (no gradio).
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator


def python_call(dotted_module: str) -> list[str]:
    """argv that runs `<module>.main()` in this same interpreter.

    e.g. python_call("rag.cli.train") → [python, "-c", "from rag.cli.train import main; main()"]
    """
    return [sys.executable, "-c", f"from {dotted_module} import main; main()"]


def stream_command(argv: list[str], env: dict[str, str] | None = None) -> Iterator[str]:
    """Run `argv`, yielding the accumulated stdout+stderr after each line, then a final
    line with the exit code."""
    proc = subprocess.Popen(
        argv,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line)
        yield "".join(log)
    proc.wait()
    log.append(f"\n[exit code {proc.returncode}]\n")
    yield "".join(log)
