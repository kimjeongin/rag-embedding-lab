"""Embedding input formatting — the asymmetric template, shared by serving AND training.

Qwen3-Embedding is asymmetric:

  - Document side: title prepended to the body, NO instruction prefix:
        "{title}\n\n{content}"   (title present)  /  "{content}"  (no title)
    Identifiers (url/domain/path) are deliberately excluded — they are filter/
    group metadata, not semantic content.
  - Query side: an instruction prefix is added:
        "Instruct: {task}\nQuery: {query}"

This module is the SINGLE definition of those rules and is kept dependency-free:
the instruction is passed IN (not read from a global config), so `core` never
depends on configuration. The default lives here as a constant for callers
(Settings, TrainingConfig) to reference.
"""
from __future__ import annotations

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_query(query: str, instruction: str) -> str:
    """Wrap a raw query in the Qwen3 instruction template."""
    return f"Instruct: {instruction}\nQuery: {query}"


def format_document(title: str | None, content: str) -> str:
    """Compose the document-side embedding input (title prepended, no prefix)."""
    if title:
        return f"{title}\n\n{content}"
    return content
