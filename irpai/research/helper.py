"""
Research Helper

Optional module that uses cues extracted from the screenshot (text
keywords, app type, headings) to run quick web searches and fetch
brief supporting context.

This is useful when the screenshot shows an error message, a UI the
LLM might not know about, or a reference to an external resource.

Install
-------
    pip install httpx beautifulsoup4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from irpai.config import ResearchConfig
from irpai.context.schema import ContextPack
from irpai.utils.logging import get_logger

log = get_logger("research")


@dataclass
class ResearchResult:
    """A single research snippet."""

    query: str
    title: str = ""
    snippet: str = ""
    url: str = ""


@dataclass
class ResearchOutput:
    """All research results for a screenshot."""

    results: list[ResearchResult] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        for r in self.results:
            if r.snippet:
                parts.append(f"[{r.query}] {r.snippet}")
        return "\n".join(parts)


class ResearchHelper:
    """
    Extracts search-worthy cues from a context pack and fetches
    brief supporting context via web search.
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.cfg = config or ResearchConfig()

    def research(self, pack: ContextPack) -> ResearchOutput:
        """
        Extract cues from *pack* and run web searches.

        Returns
        -------
        ResearchOutput
            Collected research snippets.
        """
        if not self.cfg.enabled:
            return ResearchOutput()

        queries = self._extract_queries(pack)
        if not queries:
            log.debug("No research queries extracted")
            return ResearchOutput()

        log.info("Research: %d queries to run", len(queries))

        results: list[ResearchResult] = []
        for query in queries[: self.cfg.max_results]:
            try:
                result = self._search(query)
                if result:
                    results.append(result)
            except Exception:
                log.exception("Research query failed: %s", query)

        return ResearchOutput(results=results)

    # -- query extraction ---------------------------------------------------

    def _extract_queries(self, pack: ContextPack) -> list[str]:
        """Identify search-worthy phrases from the context pack."""
        queries: list[str] = []

        # Error messages
        for region in pack.regions:
            text = region.text.lower()
            if any(kw in text for kw in ("error", "exception", "failed", "traceback", "warning")):
                # Take the first ~80 chars of the error
                clean = region.text.strip().replace("\n", " ")[:80]
                queries.append(clean)

        # URLs visible on screen
        for region in pack.regions:
            urls = re.findall(r"https?://[^\s]+", region.text)
            queries.extend(urls[:2])

        # App-specific headings
        headings = [
            r.text.strip()
            for r in pack.regions
            if r.role == "heading" and len(r.text.strip()) > 5
        ]
        queries.extend(headings[:2])

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique

    # -- search backends ----------------------------------------------------

    def _search(self, query: str) -> ResearchResult | None:
        """Run a single search query using the configured backend."""
        if self.cfg.search_provider == "duckduckgo":
            return self._search_duckduckgo(query)
        if self.cfg.search_provider == "serper":
            return self._search_serper(query)
        return None

    def _search_duckduckgo(self, query: str) -> ResearchResult | None:
        """Search via DuckDuckGo Instant Answer API (no key required)."""
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            log.warning("httpx not installed — research disabled")
            return None

        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}

        try:
            resp = httpx.get(url, params=params, timeout=self.cfg.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()

            abstract = data.get("AbstractText", "")
            if abstract:
                return ResearchResult(
                    query=query,
                    title=data.get("Heading", ""),
                    snippet=abstract[:300],
                    url=data.get("AbstractURL", ""),
                )

            # Try related topics
            related = data.get("RelatedTopics", [])
            if related and isinstance(related[0], dict):
                return ResearchResult(
                    query=query,
                    title=related[0].get("Text", "")[:100],
                    snippet=related[0].get("Text", "")[:300],
                    url=related[0].get("FirstURL", ""),
                )

        except Exception:
            log.debug("DuckDuckGo search failed for: %s", query)

        return None

    def _search_serper(self, query: str) -> ResearchResult | None:
        """Search via Serper.dev API (requires API key)."""
        if not self.cfg.api_key:
            log.warning("Serper API key not configured")
            return None

        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            log.warning("httpx not installed — research disabled")
            return None

        try:
            resp = httpx.post(
                "https://google.serper.dev/search",
                json={"q": query},
                headers={"X-API-KEY": self.cfg.api_key},
                timeout=self.cfg.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()

            organic = data.get("organic", [])
            if organic:
                top = organic[0]
                return ResearchResult(
                    query=query,
                    title=top.get("title", ""),
                    snippet=top.get("snippet", "")[:300],
                    url=top.get("link", ""),
                )
        except Exception:
            log.debug("Serper search failed for: %s", query)

        return None
