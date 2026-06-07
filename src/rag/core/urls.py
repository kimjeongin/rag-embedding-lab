"""Derive structured, filterable metadata from a page URL.

The base unit of data is the page (its URL). From the URL we derive fields used
for filtering and grouping (e.g. `site` mode):

    https://example.com/docs/x  ->  domain="example.com", path="/docs/x"

These derived fields live in the `documents.metadata` JSONB but are intentionally
kept OUT of the embedding input (see rag.core.formatting) — they are identifiers,
not semantic content.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def parse_url(url: str) -> tuple[str | None, str | None]:
    """Return (domain, path) for a URL.

    `domain` is the host only — no scheme, no port, no userinfo, lowercased
    (urlsplit.hostname). `path` is the URL path, defaulting to "/" when empty.
    """
    parts = urlsplit(url)
    domain = parts.hostname  # host only: strips scheme/port/credentials, lowercases
    path = parts.path or "/"
    return domain, path


def build_document_metadata(
    url: str | None,
    title: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge caller-supplied metadata with fields derived from the URL/title.

    Only non-null fields are written. Derived `domain`/`path` are computed from
    `url` on the server so they always stay consistent with it (and override any
    same-named keys a caller put in `extra`).
    """
    meta: dict[str, Any] = dict(extra or {})
    if url:
        domain, path = parse_url(url)
        meta["url"] = url
        meta["domain"] = domain
        meta["path"] = path
    if title:
        meta["title"] = title
    return meta
