"""NixOrb web search utility.

Searches the web using DuckDuckGo and returns formatted results
for injection into the LLM prompt context.
"""
from __future__ import annotations

import logging

import aiohttp

log = logging.getLogger(__name__)

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT = 15


async def search_formatted(
    query: str, max_results: int = 4, *, settings: object | None = None
) -> str:
    """Search the web and return formatted results.

    Args:
        query: Search query
        max_results: Maximum number of results to include

    Returns:
        Formatted search results for LLM context
    """
    try:
        results = await search(query, max_results, settings=settings)
        if not results:
            return "\n[Web search: No search results found]\n"

        formatted = ["\n[Web search results]:"]
        for i, result in enumerate(results[:max_results], 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"   {result['snippet']}")
            formatted.append(f"   URL: {result['url']}")

        return "\n".join(formatted) + "\n"

    except Exception as exc:
        log.warning("Web search failed: %s", exc)
        return f"\n[Web search: Error — {exc}]\n"


async def search(
    query: str, max_results: int = 4, *, settings: object | None = None
) -> list[dict]:
    """Search the web and return parsed results. Never raises.

    DuckDuckGo's HTML endpoint is scraped, so it rate-limits and
    occasionally changes shape. hypernix ships `search_web_non_api`, which
    covers several engines and needs no API key, so it is tried when
    scraping comes back empty — and first when `web_search_provider` says
    so.
    """
    provider = str(getattr(settings, "web_search_provider", "auto") or "auto")
    provider = provider.strip().lower()

    order: tuple[str, ...]
    if provider == "hypernix":
        order = ("hypernix", "duckduckgo")
    elif provider == "duckduckgo":
        order = ("duckduckgo",)
    else:
        order = ("duckduckgo", "hypernix")

    for name in order:
        results = await _search_with(name, query, max_results, settings)
        if results:
            if name != order[0]:
                log.info("Web search: answered by %s", name)
            return results
    return []


async def _search_with(
    provider: str, query: str, max_results: int, settings: object | None
) -> list[dict]:
    if provider == "duckduckgo":
        try:
            html = await _fetch(query)
        except Exception as exc:
            log.warning("Web search: DuckDuckGo fetch failed: %s", exc)
            return []
        return parse_results(html, max_results)

    if provider == "hypernix":
        from nixorb.utils.hypernix_client import HypernixClient

        client = HypernixClient(settings)
        if not client.supports("search_web_non_api"):
            return []
        try:
            rows = await client.search_web(query, max_results=max_results)
        except Exception as exc:
            log.warning("Web search: hypernix search failed: %s", exc)
            return []
        return [_normalise(row) for row in rows][:max_results]

    return []


def _normalise(row: dict) -> dict:
    """hypernix's result keys, mapped onto the shape callers expect."""
    return {
        "title": str(row.get("title") or row.get("name") or "").strip(),
        "snippet": str(
            row.get("snippet") or row.get("body") or row.get("description") or ""
        ).strip(),
        "url": str(row.get("url") or row.get("href") or row.get("link") or "").strip(),
    }


async def _fetch(query: str) -> str:
    """Fetch the DuckDuckGo HTML results page."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    ) as session:
        params = {"q": query}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        async with session.get(
            DUCKDUCKGO_URL, params=params, headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.text()


def parse_results(html: str, max_results: int = 4) -> list[dict]:
    """Parse DuckDuckGo's HTML results page."""
    from html.parser import HTMLParser

    class ResultParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self._current = {}
            self._in_result = False
            self._in_title = False
            self._in_snippet = False
            self._tag_stack = []

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            self._tag_stack.append(tag)

            if tag == "div" and "result" in attrs_dict.get("class", ""):
                self._in_result = True
                self._current = {}

            if self._in_result:
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_title = True
                    self._current["url"] = attrs_dict.get("href", "")

                if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self._in_snippet = True

        def handle_endtag(self, tag):
            if self._tag_stack:
                self._tag_stack.pop()

            if tag == "div" and self._in_result:
                if self._current.get("title") and self._current.get("snippet"):
                    self.results.append(self._current)
                self._in_result = False

            if tag == "a":
                self._in_title = False
                self._in_snippet = False

        def handle_data(self, data):
            if self._in_title:
                self._current["title"] = data.strip()
            elif self._in_snippet:
                self._current["snippet"] = data.strip()

    parser = ResultParser()
    parser.feed(html)
    return parser.results[:max_results]
