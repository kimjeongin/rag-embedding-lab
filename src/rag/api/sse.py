"""Server-Sent Events helper — format one SSE frame.

Shared by the streaming lab routes (training, synthetic data generation) so the wire
format lives in exactly one place.
"""
from __future__ import annotations

import json


def sse_event(event: str, data: dict) -> str:
    """One SSE frame: an ``event:`` line followed by a JSON ``data:`` line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
