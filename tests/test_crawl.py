"""Crawler pure functions — no network; trafilatura runs on inline HTML fixtures."""
from rag.datagen.crawl import (
    USER_AGENT,
    extract_page,
    finalize_pages,
    harvest_links,
    looks_like_sitemap,
    normalize_url,
    parse_sitemap,
    robots_rules,
    same_host,
)


def test_normalize_url_drops_fragment_keeps_query():
    # Query strings are identity for board/article pages (?newsId=…) — never dropped.
    assert normalize_url("HTTPS://WWW.Example.com/News/view.do?newsId=1#section") == (
        "https://www.example.com/News/view.do?newsId=1"
    )


def test_same_host_is_the_politeness_boundary():
    assert same_host("https://e.com/a", "https://e.com/")
    assert not same_host("https://cdn.e.com/a", "https://e.com/")


def test_parse_sitemap_urlset_and_index():
    urlset = (
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc> https://e.com/a </loc></url><url><loc>https://e.com/b</loc></url></urlset>"
    )
    pages, nested = parse_sitemap(urlset)
    assert pages == ["https://e.com/a", "https://e.com/b"]  # whitespace tolerated
    assert nested == []

    index = "<sitemapindex><sitemap><loc>https://e.com/s1.xml</loc></sitemap></sitemapindex>"
    pages, nested = parse_sitemap(index)
    assert pages == [] and nested == ["https://e.com/s1.xml"]
    assert looks_like_sitemap(urlset) and looks_like_sitemap(index)
    assert not looks_like_sitemap("<!doctype html><html></html>")


def test_robots_rules_disallow_and_creative_sitemap_spacing():
    robots = (
        "User-agent : Googlebot\n"          # creative spacing, seen in the wild
        "Disallow: /private/\n"
        "\n"
        "User-Agent: *\n"
        "Disallow: /admin/\n"
        "Sitemap:   https://e.com/sitemapindex.xml\n"
    )
    parser, sitemaps = robots_rules(robots)
    assert sitemaps == ["https://e.com/sitemapindex.xml"]
    assert parser.can_fetch(USER_AGENT, "https://e.com/news/1")
    assert not parser.can_fetch(USER_AGENT, "https://e.com/admin/x")


_HTML = (
    "<!doctype html><html><head><title>연차휴가 사용 안내</title>"
    '<meta name="description" content="연차휴가 신청 절차와 사용 기한 안내."></head>'
    "<body><nav>홈 메뉴 로그인</nav><main><article><h1>연차휴가 사용 안내</h1>"
    "<p>연차휴가는 입사일 기준으로 산정되며, 사용 기한은 발생일로부터 1년입니다.</p>"
    "<p>신청은 인사 시스템에서 하며, 부서장 승인 후 확정됩니다. 미사용 연차는 규정에 따라 정산됩니다.</p>"
    "<p>반차와 반반차는 별도 규정을 따르며, 자세한 내용은 인사팀에 문의하십시오.</p>"
    "</article></main></body></html>"
)


def test_extract_page_pulls_title_and_main_text():
    page = extract_page("https://e.com/notice/1", _HTML, min_chars=50)
    assert page is not None
    assert page["title"] == "연차휴가 사용 안내"
    assert "입사일 기준" in page["content"]
    assert "홈 메뉴 로그인" not in page["content"]  # boilerplate nav stays out


def test_extract_page_rejects_thin_pages():
    # min_chars is the nav/list-page filter: not enough main text → not a corpus page.
    assert extract_page("https://e.com/notice/1", _HTML, min_chars=100_000) is None


def test_finalize_pages_drops_boilerplate_descriptions_and_caps():
    pages = [
        {"url": f"u{i}", "title": f"t{i}", "description": "사이트 공통 설명문", "content": "본문 " * 60}
        for i in range(12)
    ]
    pages.append(
        {"url": "u-x", "title": "tx", "description": "이 페이지만의 요약", "content": "본문내용 " * 60}
    )
    out = finalize_pages(pages, max_chars=100)
    # a description shared by 12/13 pages is template noise → removed everywhere
    assert all(p["description"] is None for p in out[:12])
    assert all("사이트 공통 설명문" not in p["content"] for p in out[:12])
    # the unique one survives and leads the page text (FirstP: summary first)
    assert out[-1]["description"] == "이 페이지만의 요약"
    assert out[-1]["content"].startswith("이 페이지만의 요약")
    assert all(len(p["content"]) <= 100 for p in out)


def test_finalize_pages_skips_fold_when_body_opens_with_description():
    pages = [{"url": "u", "title": "t", "description": "첫 문장입니다", "content": "첫 문장입니다. 그리고 둘째."}]
    out = finalize_pages(pages, max_chars=1000)
    assert out[0]["content"] == "첫 문장입니다. 그리고 둘째."  # no duplicated prefix


def test_harvest_links_filters_offsite_binaries_and_schemes():
    html = (
        '<a href="/a?id=1">x</a> <a href="https://other.com/b">y</a> '
        '<a href="files/doc.hwp">z</a> <a href="mailto:a@b.com">m</a> <a href="/c#frag">c</a>'
    )
    links = harvest_links(html, "https://e.com/base/")
    assert "https://e.com/a?id=1" in links
    assert "https://e.com/c" in links               # fragment normalized away
    assert all("other.com" not in link for link in links)
    assert all(not link.endswith(".hwp") for link in links)
