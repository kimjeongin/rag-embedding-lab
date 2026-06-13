"""Crawl a public website into the lab's corpus (page = retrieval unit).

The lab's real target is an internal-site search model; its PoC stand-in is any
public site. This module turns one into ``data/corpus.jsonl`` records that the rest
of the pipeline (synthetic queries → train → eval) consumes unchanged:

    {"url": ..., "title": ..., "description": ..., "content": ...}

Decisions that shape the corpus:

  - **Page-level units** (the production index unit). ``content`` is the page's
    embedding text: meta description + extracted main text, capped at ``max_chars``
    — the "FirstP" strategy. Long-document ranking studies keep finding that
    title + the first ~500 tokens is a near-unbeatable page representation,
    because web pages put the substance first.
  - **Boilerplate descriptions are dropped.** Many sites repeat one site-wide meta
    description on every page; a description string appearing on >10% of pages is
    template noise, not page signal, so it's removed from all of them.
  - **Politeness.** robots.txt is honored, fetches are sequential with a fixed
    delay, and the page budget is a hard cap. Discovery is sitemap-first (index
    files followed; the start URL may itself BE a sitemap, to crawl one section
    of a large site); same-host BFS over page links is the fallback.

trafilatura does the per-page extraction (title/description/main text); the crawl
loop stays ours so the politeness and budget rules are visible here, not buried in
a library.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import urllib.robotparser
from collections import Counter
from collections.abc import AsyncIterator
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

USER_AGENT = "rag-lab-poc/0.1 (embedding fine-tuning PoC; sitemap-first, rate-limited)"
_FETCH_TIMEOUT = 15.0
_LOC = re.compile(r"<loc>\s*([^<\s][^<]*?)\s*</loc>")
_HREF = re.compile(r"""href\s*=\s*["']([^"'<>\s]+)""", re.IGNORECASE)
_SITEMAP_LINE = re.compile(r"(?i)^\s*sitemap\s*:\s*(\S+)")
# Link targets that can't become corpus pages (binary downloads, non-http schemes).
_SKIP_EXTENSIONS = (".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx", ".zip",
                    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".mp3", ".css", ".js")


# ── pure helpers (unit-testable, no IO) ─────────────────────────────────────────
def normalize_url(url: str) -> str:
    """Canonical form for dedupe: scheme/host lowercased, fragment dropped. The query
    string is KEPT — board/article pages live in it (e.g. ?newsId=148966356)."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def same_host(url: str, root: str) -> bool:
    """True if `url` is on the crawl root's host (politeness boundary: never leave it)."""
    return urlsplit(url).netloc.lower() == urlsplit(root).netloc.lower()


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """(page_urls, nested_sitemap_urls) from one sitemap document.

    A <sitemapindex> lists more sitemaps; a <urlset> lists pages. Parsed with a
    regex over <loc> — sitemaps in the wild are mechanical enough that this is
    more robust than namespace-sensitive XML parsing.
    """
    locs = _LOC.findall(xml_text)
    if "<sitemapindex" in xml_text:
        return [], locs
    return locs, []


def robots_rules(robots_txt: str) -> tuple[urllib.robotparser.RobotFileParser, list[str]]:
    """(rule parser, sitemap urls) from a robots.txt body. The Sitemap: lines are
    extracted by hand — sites write them with creative spacing the parser misses."""
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(robots_txt.splitlines())
    sitemaps = [m.group(1) for line in robots_txt.splitlines() if (m := _SITEMAP_LINE.match(line))]
    return parser, sitemaps


def looks_like_sitemap(text: str) -> bool:
    """True if a fetched body is a sitemap document (so a start URL can BE a sitemap)."""
    head = text[:500]
    return "<urlset" in head or "<sitemapindex" in head


def _tidy(text: str) -> str:
    """Collapse extraction whitespace: strip every line, drop the empty ones."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_page(url: str, html: str, min_chars: int) -> dict | None:
    """One fetched page → a corpus record, or None (no main content / too thin to be
    a retrieval unit — nav and list pages die here)."""
    import json

    import trafilatura

    raw = trafilatura.extract(html, output_format="json", with_metadata=True, url=url)
    if not raw:
        return None
    data = json.loads(raw)
    body = _tidy(data.get("text") or "")
    if len(body) < min_chars:
        return None
    return {
        "url": url,
        "title": (data.get("title") or "").strip() or None,
        "description": _tidy(data.get("description") or "") or None,
        "content": body,
    }


def finalize_pages(pages: list[dict], max_chars: int) -> list[dict]:
    """Site-wide boilerplate descriptions out, then description+body folded into the
    capped ``content`` (the page's embedding text). ``description`` stays as its own
    field so later experiments can re-compose the page text without recrawling."""
    counts = Counter(p["description"] for p in pages if p["description"])
    boilerplate = {d for d, n in counts.items() if len(pages) >= 10 and n > len(pages) * 0.1}
    out = []
    for page in pages:
        desc = page["description"]
        if desc in boilerplate:
            desc = None
        body = page["content"]
        # Skip the fold when the body already opens with the description (common when
        # the meta description is just the first sentence).
        text = f"{desc}\n\n{body}" if desc and not body.startswith(desc[:40]) else body
        out.append({**page, "description": desc, "content": text[:max_chars].rstrip()})
    return out


def harvest_links(html: str, base_url: str) -> list[str]:
    """Same-host page links from raw HTML (the BFS fallback's frontier)."""
    found = []
    for href in _HREF.findall(html):
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if not absolute.startswith(("http://", "https://")):
            continue
        if urlsplit(absolute).path.lower().endswith(_SKIP_EXTENSIONS):
            continue
        if same_host(absolute, base_url):
            found.append(normalize_url(absolute))
    return found


# ── the crawl loop ──────────────────────────────────────────────────────────────
async def _get_text(http: httpx.AsyncClient, url: str) -> str | None:
    try:
        resp = await http.get(url)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError:
        return None


async def _discover(
    http: httpx.AsyncClient, start_url: str, robots_sitemaps: list[str], delay: float
) -> list[str]:
    """Page URLs from sitemaps: the start URL itself if it is one, else the ones
    robots.txt declares, else /sitemap.xml. Index files are followed one level.
    Empty result = no sitemap anywhere → caller falls back to BFS."""
    first = await _get_text(http, start_url)
    if first and looks_like_sitemap(first):
        candidates = [start_url]
    else:
        candidates = robots_sitemaps or [urljoin(start_url, "/sitemap.xml")]

    urls: list[str] = []
    seen_maps: set[str] = set()
    queue = list(candidates)
    while queue:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen_maps:
            continue
        seen_maps.add(sitemap_url)
        text = first if sitemap_url == start_url and first else await _get_text(http, sitemap_url)
        await asyncio.sleep(delay)
        if not text or not looks_like_sitemap(text):
            continue
        page_urls, nested = parse_sitemap(text)
        urls.extend(page_urls)
        queue.extend(nested)
    return urls


async def crawl_stream(
    start_url: str,
    max_pages: int = 300,
    delay: float = 0.4,
    min_chars: int = 200,
    max_chars: int = 2000,
) -> AsyncIterator[dict]:
    """Crawl one site, yielding progress events as it goes.

    Events: ``start`` (discovery mode + frontier size), one ``page`` per kept page,
    and a terminal ``done`` carrying the finalized corpus records. Sequential by
    design — a polite crawler is slower than a swarm, and the corpus is built once.
    """
    root = normalize_url(start_url)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True, headers=headers) as http:
        robots_txt = await _get_text(http, urljoin(root, "/robots.txt")) or ""
        rules, robots_sitemaps = robots_rules(robots_txt)

        frontier = await _discover(http, root, robots_sitemaps, delay)
        bfs = not frontier
        if bfs:
            frontier = [root]
        yield {"event": "start", "mode": "bfs" if bfs else "sitemap",
               "discovered": len(frontier), "max_pages": max_pages}

        seen: set[str] = set()
        content_hashes: set[str] = set()
        pages: list[dict] = []
        fetched = skipped = 0

        while frontier and len(pages) < max_pages:
            url = normalize_url(urljoin(root, frontier.pop(0)))
            if url in seen or not same_host(url, root):
                continue
            seen.add(url)
            if not rules.can_fetch(USER_AGENT, url):
                skipped += 1
                continue

            try:
                resp = await http.get(url)
                resp.raise_for_status()
            except httpx.HTTPError:
                skipped += 1
                continue
            fetched += 1
            await asyncio.sleep(delay)
            if "html" not in resp.headers.get("content-type", ""):
                skipped += 1
                continue

            if bfs:
                frontier.extend(u for u in harvest_links(resp.text, url) if u not in seen)

            page = extract_page(url, resp.text, min_chars)
            if page is None:
                skipped += 1
                continue
            digest = hashlib.sha256(page["content"].encode()).hexdigest()[:16]
            if digest in content_hashes:  # boards re-serve one article under many URLs
                skipped += 1
                continue
            content_hashes.add(digest)
            pages.append(page)
            yield {"event": "page", "done": len(pages), "total": max_pages,
                   "url": url, "title": page["title"], "chars": len(page["content"])}

    yield {"event": "done", "pages": finalize_pages(pages, max_chars),
           "fetched": fetched, "skipped": skipped}
