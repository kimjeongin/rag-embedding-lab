"""Asymmetric embedding-input formatting."""
from rag.core.formatting import format_document, format_query


def test_format_query_wraps_with_instruction():
    assert format_query("hi", "do X") == "Instruct: do X\nQuery: hi"


def test_format_document_prepends_title():
    assert format_document("Title", "body") == "Title\n\nbody"


def test_format_document_without_title_is_body_only():
    assert format_document(None, "body") == "body"
    assert format_document("", "body") == "body"
