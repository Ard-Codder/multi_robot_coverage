import json

import pytest

from agent.research_tools import search_arxiv, search_habr_rss, search_semantic_scholar


def test_search_arxiv_structure() -> None:
    s = search_arxiv("robot", max_results=1)
    data = json.loads(s)
    if "error" in data:
        pytest.skip(data["error"])
    assert "items" in data
    assert isinstance(data["items"], list)


def test_search_arxiv_json_always_dict() -> None:
    s = search_arxiv("x", max_results=1)
    data = json.loads(s)
    assert isinstance(data, dict)


def test_semantic_scholar_structure() -> None:
    s = search_semantic_scholar("multi robot", max_results=2)
    data = json.loads(s)
    if "error" in data:
        pytest.skip(str(data.get("error")))
    assert "items" in data


def test_habr_rss_structure() -> None:
    s = search_habr_rss("python", max_items=2)
    data = json.loads(s)
    if "error" in data:
        pytest.skip(data["error"])
    assert "items" in data
