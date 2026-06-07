"""URL parsing + metadata derivation."""
from rag.core.urls import build_document_metadata, parse_url


def test_parse_url_host_and_path():
    assert parse_url("https://example.com/docs/x") == ("example.com", "/docs/x")


def test_parse_url_strips_port_userinfo_and_lowercases():
    domain, path = parse_url("https://user:pw@Example.com:8443/a/b?q=1#frag")
    assert domain == "example.com"
    assert path == "/a/b"


def test_parse_url_empty_path_defaults_to_root():
    assert parse_url("https://example.com") == ("example.com", "/")


def test_build_metadata_derives_and_merges_extras():
    meta = build_document_metadata("https://e.com/p", "Title", {"k": "v"})
    assert meta == {"k": "v", "url": "https://e.com/p", "domain": "e.com",
                    "path": "/p", "title": "Title"}


def test_build_metadata_derived_overrides_extra():
    meta = build_document_metadata("https://e.com/p", None, {"domain": "WRONG"})
    assert meta["domain"] == "e.com"


def test_build_metadata_without_url():
    assert build_document_metadata(None, "T", {}) == {"title": "T"}
