"""Web retrieval helpers.

Botasaurus is optional. The first version uses existing scholarly API helpers
and falls back to simple URL scraping for user-provided links.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from dataclasses import dataclass

from agent.research_tools import search_arxiv, search_semantic_scholar
from diploma_agent.retrievers.local_docs import SourceChunk


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str


def clean_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_items(raw: str) -> list[WebSearchResult]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    source = data.get("source", "web")
    items = []
    for item in data.get("items", []) or []:
        items.append(
            WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url") or item.get("link", ""),
                snippet=item.get("abstract") or item.get("summary") or item.get("snippet", ""),
                source=source,
            )
        )
    return items


def search(query: str, max_results: int = 6) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    results.extend(_parse_items(search_arxiv(query, max_results=max_results)))
    results.extend(_parse_items(search_semantic_scholar(query, max_results=max_results)))
    return results[: max_results * 2]


def scrape(urls: list[str], timeout: float = 20.0) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DiplomaAgent/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        cleaned = clean_html(raw)
        if cleaned:
            chunks.append(SourceChunk(url, "scraped page", cleaned[:3000]))
    return chunks


def search_as_chunks(query: str, max_results: int = 6) -> list[SourceChunk]:
    return [
        SourceChunk(item.url or item.source, item.title or "web result", item.snippet, 1.0)
        for item in search(query, max_results=max_results)
        if item.snippet
    ]
