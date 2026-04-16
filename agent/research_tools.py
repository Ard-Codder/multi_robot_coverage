"""
Поиск материалов для обзора: arXiv (официальный API), Semantic Scholar (публичный API), Habr (RSS).

Используйте умеренную частоту запросов; для продакшена — кэш и задержки.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

USER_AGENT = "CoverageResearchBot/1.0 (diploma research; +https://arxiv.org/help/api)"
SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HABR_RSS_SEARCH = "https://habr.com/ru/rss/search/"
ARXIV_API = "https://export.arxiv.org/api/query"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = _ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def search_arxiv(query: str, max_results: int = 8) -> str:
    """
    Поиск препринтов через официальный arXiv API (Atom).
    https://arxiv.org/help/api/user-manual
    """
    q = urllib.parse.quote(f"all:{query.strip()}")
    url = (
        f"{ARXIV_API}?search_query={q}"
        f"&start=0&max_results={max(1, min(max_results, 30))}&sortBy=relevance&sortOrder=descending"
    )
    try:
        raw = _http_get(url)
    except urllib.error.HTTPError as e:
        return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)
    except OSError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    items: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        id_el = entry.find("atom:id", ns)
        summ_el = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        title = (title_el.text or "").strip().replace("\n", " ")
        link = (id_el.text or "").strip()
        summary = (summ_el.text or "").strip().replace("\n", " ")[:600] if summ_el is not None else ""
        pub = (published.text or "").strip() if published is not None else ""
        items.append(
            {
                "title": title,
                "link": link,
                "published": pub,
                "summary": summary + ("..." if summ_el is not None and len(summ_el.text or "") > 600 else ""),
            }
        )
    return json.dumps({"source": "arxiv", "query": query, "items": items}, indent=2, ensure_ascii=False)


def search_semantic_scholar(query: str, max_results: int = 8) -> str:
    """
    Поиск статей через публичный Graph API Semantic Scholar (без ключа, лимиты по документации).
    """
    params = urllib.parse.urlencode(
        {
            "query": query.strip(),
            "limit": max(1, min(max_results, 20)),
            "fields": "title,authors,year,url,abstract",
        }
    )
    url = f"{SCHOLAR_URL}?{params}"
    try:
        raw = _http_get(url, timeout=25.0)
    except urllib.error.HTTPError as e:
        return json.dumps({"error": str(e), "hint": "Возможен rate limit — повторите позже"}, ensure_ascii=False)
    except OSError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    data = json.loads(raw.decode("utf-8"))
    items: List[Dict[str, Any]] = []
    for p in data.get("data") or []:
        authors = p.get("authors") or []
        names = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            names += "…"
        abstract = (p.get("abstract") or "")[:500]
        items.append(
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "authors": names,
                "url": p.get("url", ""),
                "abstract": abstract + ("..." if len(p.get("abstract") or "") > 500 else ""),
            }
        )
    return json.dumps(
        {"source": "semantic_scholar", "query": query, "total": data.get("total"), "items": items},
        indent=2,
        ensure_ascii=False,
    )


def search_habr_rss(query: str, max_items: int = 10) -> str:
    """
    Поиск по Habr через RSS (публичная лента, без авторизации).
    """
    q = urllib.parse.quote(query.strip())
    url = f"{HABR_RSS_SEARCH}?q={q}&order_by=date"
    try:
        raw = _http_get(url)
    except urllib.error.HTTPError as e:
        return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)
    except OSError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    root = ET.fromstring(raw)
    channel = root.find("channel")
    if channel is None:
        return json.dumps({"error": "Неверный RSS", "source": "habr"}, ensure_ascii=False)

    items: List[Dict[str, str]] = []
    for item in channel.findall("item")[: max(1, min(max_items, 30))]:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        pub = (pub_el.text or "").strip() if pub_el is not None else ""
        # убираем простой HTML из description
        desc = desc.replace("<br/>", " ").replace("<br>", " ")[:400]
        items.append({"title": title, "link": link, "published": pub, "snippet": desc})
    return json.dumps({"source": "habr_rss", "query": query, "items": items}, indent=2, ensure_ascii=False)
