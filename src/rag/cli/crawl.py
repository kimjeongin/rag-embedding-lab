"""`rag-crawl` — crawl a public site into the corpus file (rag.datagen.crawl).

The PoC corpus source: point it at a site root — or directly at a sitemap.xml to
crawl one section of a large site — and the extracted pages become the corpus the
synthetic query generator reads:

    uv run rag-crawl https://www.korea.kr/sitemap_policy.xml

Env: CRAWL_URL (alternative to the argument), CRAWL_MAX_PAGES (300), CRAWL_DELAY
(0.4s between fetches), CRAWL_MIN_CHARS (200 — thinner pages are skipped),
CRAWL_MAX_CHARS (2000 — page text cap, the FirstP cut), CORPUS_FILE (output,
default data/corpus.jsonl).
"""
from __future__ import annotations

import asyncio
import os
import sys

from rag.datagen.crawl import crawl_stream
from rag.dataset import write_jsonl


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CRAWL_URL", "")
    if not url:
        raise SystemExit("[crawl] 대상 URL이 필요합니다 — `rag-crawl <url>` 또는 CRAWL_URL=<url>")
    corpus_file = os.getenv("CORPUS_FILE", "data/corpus.jsonl")
    max_pages = int(os.getenv("CRAWL_MAX_PAGES", "300"))
    delay = float(os.getenv("CRAWL_DELAY", "0.4"))
    min_chars = int(os.getenv("CRAWL_MIN_CHARS", "200"))
    max_chars = int(os.getenv("CRAWL_MAX_CHARS", "2000"))

    async def _run() -> dict:
        async for event in crawl_stream(url, max_pages, delay, min_chars, max_chars):
            if event["event"] == "start":
                print(f"[crawl] mode={event['mode']} discovered={event['discovered']} "
                      f"max_pages={event['max_pages']} delay={delay}s")
            elif event["event"] == "page":
                print(f"[crawl] {event['done']}/{event['total']} {event['title'] or event['url']}")
            elif event["event"] == "done":
                return event
        return {"pages": [], "fetched": 0, "skipped": 0}

    result = asyncio.run(_run())
    pages = result["pages"]
    if not pages:
        raise SystemExit("[crawl] 수집된 페이지가 없습니다 — URL, robots.txt, CRAWL_MIN_CHARS를 확인하세요")
    write_jsonl(corpus_file, pages)
    print(f"[crawl] wrote {corpus_file} ({len(pages)} pages · "
          f"fetched={result['fetched']} skipped={result['skipped']})")
