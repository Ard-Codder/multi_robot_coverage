"""Keyword retrieval over local project documents and result artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config

WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]{3,}", re.UNICODE)

DEFAULT_PATHS = (
    "README.md",
    "docs/DIPLOMA_CHAPTER_DRAFT.md",
    "docs/DIPLOMA_RESEARCH_PACKAGE.md",
    "docs/DIPLOMA_EXPERIMENTS_AND_METRICS.md",
    "docs/DIPLOMA_VLA_POSITION.md",
    "docs/EXPERIMENTS_OVERVIEW.md",
    "docs/RESULTS_OVERVIEW.md",
    "docs/METHOD_GROUPS.md",
    "coverage_lab/result_contract.md",
    "results/lab/presentation_report/report_cards.md",
    "results/lab/presentation_report/report_cards.csv",
    "results/lab/presentation_static/summary.csv",
    "results/lab/presentation_dynamic/summary.csv",
    "results/lab/presentation_dynamic_ml_rl/summary.csv",
    "results/lab/presentation_dynamic_rl/summary.csv",
)


@dataclass(frozen=True)
class SourceChunk:
    source: str
    title: str
    text: str
    score: float = 0.0

    def as_prompt_block(self) -> str:
        return f"[{self.source} :: {self.title}]\n{self.text.strip()}"


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def _read_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ""
    head, *body = rows
    lines = [", ".join(head)]
    for row in body[:40]:
        lines.append(", ".join(row))
    if len(body) > 40:
        lines.append(f"... ({len(body) - 40} more rows)")
    return "\n".join(lines)


def _read_json(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(raw, indent=2, ensure_ascii=False)[:6000]


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() == ".json":
        return _read_json(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _markdown_chunks(rel_path: str, text: str, max_chars: int = 2600) -> list[SourceChunk]:
    headings = list(re.finditer(r"^(#{1,4})\s+(.+)$", text, flags=re.MULTILINE))
    if not headings:
        return _fixed_chunks(rel_path, Path(rel_path).name, text, max_chars)
    chunks: list[SourceChunk] = []
    for idx, heading in enumerate(headings):
        start = heading.start()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        title = heading.group(2).strip()
        chunk_text = text[start:end].strip()
        chunks.extend(_fixed_chunks(rel_path, title, chunk_text, max_chars))
    return chunks


def _fixed_chunks(rel_path: str, title: str, text: str, max_chars: int = 2600) -> list[SourceChunk]:
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [SourceChunk(rel_path, title, clean)]
    chunks = []
    for idx in range(0, len(clean), max_chars):
        part = clean[idx : idx + max_chars].strip()
        if part:
            chunks.append(SourceChunk(rel_path, f"{title} #{idx // max_chars + 1}", part))
    return chunks


class LocalDocsRetriever:
    def __init__(self, root: Path | None = None, paths: tuple[str, ...] = DEFAULT_PATHS) -> None:
        self.root = root or config.ROOT
        self.paths = paths
        self._chunks: list[SourceChunk] | None = None

    def chunks(self) -> list[SourceChunk]:
        if self._chunks is None:
            loaded: list[SourceChunk] = []
            for rel in self.paths:
                path = self.root / rel
                if not path.exists():
                    continue
                try:
                    text = _read_text(path)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if path.suffix.lower() == ".md":
                    loaded.extend(_markdown_chunks(rel, text))
                else:
                    loaded.extend(_fixed_chunks(rel, path.name, text))
            self._chunks = loaded
        return self._chunks

    def search(self, query: str, top_k: int = 7) -> list[SourceChunk]:
        query_terms = tokenize(query)
        if not query_terms:
            return self.chunks()[:top_k]
        scored: list[SourceChunk] = []
        for chunk in self.chunks():
            terms = tokenize(chunk.title + "\n" + chunk.text)
            overlap = query_terms & terms
            if not overlap:
                continue
            coverage = len(overlap) / math.sqrt(max(len(query_terms), 1))
            title_bonus = len(query_terms & tokenize(chunk.title)) * 0.5
            scored.append(
                SourceChunk(
                    source=chunk.source,
                    title=chunk.title,
                    text=chunk.text,
                    score=coverage + title_bonus,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
