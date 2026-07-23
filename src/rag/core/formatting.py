"""Embedding input formatting — the per-model templates, shared by serving AND training.

Different embedding models want different input formats, and getting this wrong is a
**silent** failure: vectors still come out, search still ranks, the numbers are just
quietly worse. Nothing raises. So the format is a per-model `ModelProfile`, defined
here once and threaded through train / eval / serve identically — train/inference
parity is the invariant this module exists to protect.

  Qwen3-Embedding — instruction-prefixed query, title prepended to the document:
        query: "Instruct: {task}\\nQuery: {query}"
        doc:   "{title}\\n\\n{content}"
  Nemotron-3-Embed — short literal prefixes on both sides (no instruction):
        query: "query: {query}"
        doc:   "passage: {title}\\n\\n{content}"

Identifiers (url/domain/path) are deliberately excluded from the document side in
every profile — they are filter/group metadata, not semantic content.

This module is the SINGLE definition of those rules and is kept dependency-free: the
instruction and the profile are passed IN (never read from a global config or the
filesystem), so `core` never depends on configuration. Deciding WHICH profile a given
model wants is `rag.modelprofile`'s job — that's the part that touches env and disk.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """How ONE model family wants its query/document text shaped.

    Templates are ``str.format`` patterns: the query side may use ``{instruction}``
    and ``{query}``, the document side ``{title}`` and ``{content}``.
    ``uses_instruction`` says whether the instruction text reaches the model at all —
    a profile that ignores it (Nemotron) must not leave callers believing it matters.
    """

    name: str
    query_template: str
    doc_template: str            # used when the document has a title
    doc_template_untitled: str   # used when it doesn't
    uses_instruction: bool


# Qwen3-Embedding: asymmetric, instruction on the query side, last-token pooling.
QWEN3 = ModelProfile(
    name="qwen3",
    query_template="Instruct: {instruction}\nQuery: {query}",
    doc_template="{title}\n\n{content}",
    doc_template_untitled="{content}",
    uses_instruction=True,
)

# Nemotron-3-Embed: literal "query: " / "passage: " prefixes, average pooling, 2048-d.
# (sentence-transformers ships these as saved prompts behind encode_query/
# encode_document; we spell them out so TRAINING uses the exact same strings.)
NEMOTRON3 = ModelProfile(
    name="nemotron3",
    query_template="query: {query}",
    doc_template="passage: {title}\n\n{content}",
    doc_template_untitled="passage: {content}",
    uses_instruction=False,
)

# No prefixes at all — for symmetric models, or to measure what the prefixes are worth.
PLAIN = ModelProfile(
    name="plain",
    query_template="{query}",
    doc_template="{title}\n\n{content}",
    doc_template_untitled="{content}",
    uses_instruction=False,
)

PROFILES: dict[str, ModelProfile] = {p.name: p for p in (QWEN3, NEMOTRON3, PLAIN)}

# The lab's historical default. Kept as the fallback so every run recorded before
# profiles existed stays reproducible bit-for-bit.
DEFAULT_PROFILE = QWEN3

# Substring → profile, first match wins. Checked against the model name/path lowercased.
_NAME_RULES: tuple[tuple[str, ModelProfile], ...] = (
    ("nemotron", NEMOTRON3),
    ("qwen", QWEN3),
)


def profile_for_name(model: str) -> ModelProfile | None:
    """The profile a model NAME implies, or None when nothing matches.

    Returning None (rather than a default) is deliberate: the caller decides whether
    an unrecognised model is a fallback-with-warning or a hard error.
    """
    lowered = model.lower()
    for needle, profile in _NAME_RULES:
        if needle in lowered:
            return profile
    return None


def format_query(query: str, instruction: str, profile: ModelProfile = DEFAULT_PROFILE) -> str:
    """Compose the query-side embedding input for ``profile``."""
    return profile.query_template.format(instruction=instruction, query=query)


def format_document(
    title: str | None, content: str, profile: ModelProfile = DEFAULT_PROFILE
) -> str:
    """Compose the document-side embedding input for ``profile``."""
    template = profile.doc_template if title else profile.doc_template_untitled
    return template.format(title=title or "", content=content)
