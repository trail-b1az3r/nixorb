"""tests/test_web_search.py — Web search utility tests."""
from __future__ import annotations

from unittest.mock import patch

_RESULT_HTML = """
<div class="result">
  <a class="result__a" href="https://example.com">Example Title</a>
  <a class="result__snippet">This is a snippet about the topic.</a>
</div>
"""


def test_parse_results_extracts_fields():
    from nixorb.utils.web_search import parse_results

    results = parse_results(_RESULT_HTML, max_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "Example Title"
    assert results[0]["url"] == "https://example.com"
    assert "snippet" in results[0]["snippet"]


def test_parse_results_honours_max():
    from nixorb.utils.web_search import parse_results

    results = parse_results(_RESULT_HTML * 5, max_results=2)
    assert len(results) == 2


def test_parse_results_on_junk_html():
    from nixorb.utils.web_search import parse_results

    assert parse_results("<html><body>nothing here</body></html>") == []


async def test_search_returns_list():
    from nixorb.utils.web_search import search

    with patch("nixorb.utils.web_search._fetch", return_value=_RESULT_HTML):
        results = await search("test query", max_results=3)

    assert isinstance(results, list)
    assert results[0]["title"] == "Example Title"


async def test_search_fails_gracefully():
    """A network error must degrade to no results, never propagate."""
    from nixorb.utils.web_search import search

    async def _fail(_query):
        raise OSError("network down")

    with patch("nixorb.utils.web_search._fetch", _fail):
        results = await search("anything")

    assert results == []


async def test_search_formatted_no_results():
    from nixorb.utils.web_search import search_formatted

    async def _none(_query, _max, **_kwargs):
        return []

    with patch("nixorb.utils.web_search.search", _none):
        result = await search_formatted("xyz")

    assert "No search results" in result


async def test_search_formatted_includes_titles_and_urls():
    from nixorb.utils.web_search import search_formatted

    async def _one(_query, _max, **_kwargs):
        return [{"title": "T", "snippet": "S", "url": "https://u"}]

    with patch("nixorb.utils.web_search.search", _one):
        result = await search_formatted("xyz")

    assert "T" in result and "https://u" in result


async def test_wants_web_detection():
    from nixorb.main import _wants_web

    assert _wants_web("what is the current price of bitcoin")
    assert _wants_web("search for Arch Linux news")
    assert _wants_web("who is Linus Torvalds")
    assert not _wants_web("open my terminal")


async def test_wants_screen_detection():
    from nixorb.main import _wants_screen

    assert _wants_screen("what am I looking at")
    assert _wants_screen("what's on my screen")
    assert _wants_screen("see my screen")
    assert not _wants_screen("play some music")


# ── providers ────────────────────────────────────────────────────── #

class TestSearchProviders:
    """DuckDuckGo's HTML endpoint is scraped, so it rate-limits and
    changes shape. hypernix's search_web_non_api covers several engines
    and needs no API key, so it stands behind (or in front of) it."""

    async def test_duckduckgo_is_tried_first_by_default(self, monkeypatch):
        from nixorb.utils import web_search

        called = []

        async def fake(provider, query, max_results, settings):
            called.append(provider)
            return [{"title": "t", "snippet": "s", "url": "u"}]

        monkeypatch.setattr(web_search, "_search_with", fake)
        await web_search.search("anything")
        assert called == ["duckduckgo"]

    async def test_hypernix_answers_when_scraping_comes_back_empty(
        self, monkeypatch
    ):
        from nixorb.utils import web_search

        called = []

        async def fake(provider, query, max_results, settings):
            called.append(provider)
            if provider == "duckduckgo":
                return []
            return [{"title": "t", "snippet": "s", "url": "u"}]

        monkeypatch.setattr(web_search, "_search_with", fake)
        results = await web_search.search("anything")
        assert called == ["duckduckgo", "hypernix"]
        assert results[0]["title"] == "t"

    async def test_the_provider_can_be_pinned(self, monkeypatch):
        from nixorb.settings import Settings
        from nixorb.utils import web_search

        called = []

        async def fake(provider, query, max_results, settings):
            called.append(provider)
            return [{"title": "t", "snippet": "s", "url": "u"}]

        monkeypatch.setattr(web_search, "_search_with", fake)
        await web_search.search(
            "x", settings=Settings(web_search_provider="hypernix")
        )
        assert called == ["hypernix"]

        called.clear()
        await web_search.search(
            "x", settings=Settings(web_search_provider="duckduckgo")
        )
        assert called == ["duckduckgo"]

    async def test_both_failing_returns_nothing_rather_than_raising(
        self, monkeypatch
    ):
        from nixorb.utils import web_search

        async def nothing(provider, query, max_results, settings):
            return []

        monkeypatch.setattr(web_search, "_search_with", nothing)
        assert await web_search.search("x") == []

    def test_hypernix_result_keys_are_normalised(self):
        from nixorb.utils.web_search import _normalise

        assert _normalise({"title": "T", "body": "B", "href": "U"}) == {
            "title": "T", "snippet": "B", "url": "U",
        }
        assert _normalise({"name": "T", "description": "D", "link": "L"}) == {
            "title": "T", "snippet": "D", "url": "L",
        }

    async def test_a_hypernix_failure_does_not_escape(self, monkeypatch):
        from nixorb.utils import web_search

        class _Boom:
            def __init__(self, settings=None):
                pass

            def supports(self, name):
                return True

            async def search_web(self, *a, **k):
                raise RuntimeError("network down")

        monkeypatch.setattr(
            "nixorb.utils.hypernix_client.HypernixClient", _Boom
        )
        assert await web_search._search_with("hypernix", "q", 4, None) == []
