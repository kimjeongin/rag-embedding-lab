"""Request-scoped dependencies.

``Settings`` is built once at the composition root (``rag.api.app.create_app``) and
stashed on ``app.state``; this accessor hands it to the lab routes via ``Depends`` so
handlers never reach into ``app.state`` directly.
"""
from __future__ import annotations

from fastapi import Request

from rag.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
