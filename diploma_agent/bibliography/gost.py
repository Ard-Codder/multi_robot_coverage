"""GOST-like bibliography formatting.

This is a pragmatic formatter for bachelor thesis drafts. Department-specific
requirements can later replace these rules with an exact template.
"""

from __future__ import annotations

from pathlib import Path

from diploma_agent.sources.source_store import SourceRecord, SourceStore
from diploma_agent.state import ensure_workspace


def build_bibliography_markdown(records: list[SourceRecord]) -> str:
    lines = ["# Список использованных источников", ""]
    for idx, record in enumerate(sorted(records, key=lambda item: (item.source_type, item.title.lower())), start=1):
        lines.append(f"{idx}. {_format_record(record)}")
    return "\n".join(lines).strip() + "\n"


def save_bibliography(workspace: Path | None = None) -> Path:
    root = ensure_workspace(workspace)
    store = SourceStore(root)
    records = _curated_records(list(store.load().values()))
    for record in _default_project_sources():
        if all(existing.id != record.id for existing in records):
            records.append(record)
            store.upsert(record)
    path = root / "build" / "bibliography.md"
    path.write_text(build_bibliography_markdown(records), encoding="utf-8")
    return path


def _format_record(record: SourceRecord) -> str:
    authors = ", ".join(record.authors)
    prefix = f"{authors} " if authors else ""
    year = f" {record.year}." if record.year else ""
    if record.source_type == "project":
        url = f" — {record.url}" if record.url else ""
        return f"{prefix}{record.title}{url}.{year}"
    if record.source_type == "paper":
        container = f" // {record.container}" if record.container else ""
        url = f" URL: {record.url}" if record.url else ""
        return f"{prefix}{record.title}{container}.{year}{url}"
    accessed = f" (дата обращения: {record.accessed_at[:10]})" if record.accessed_at else ""
    return f"{prefix}{record.title} [Электронный ресурс]. URL: {record.url}{accessed}."


def _curated_records(records: list[SourceRecord]) -> list[SourceRecord]:
    allowed_project_ids = {"project-readme", "project-results", "project-methods"}
    filtered = []
    seen = set()
    for record in records:
        if record.source_type == "project" and record.id not in allowed_project_ids:
            continue
        key = (record.title.strip().lower(), record.url.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append(record)
    return filtered


def _default_project_sources() -> list[SourceRecord]:
    return [
        SourceRecord(
            id="project-readme",
            title="Multi-Robot Coverage Research",
            url="README.md",
            source_type="project",
            year="2026",
        ),
        SourceRecord(
            id="project-results",
            title="Результаты экспериментов multi-robot coverage",
            url="results/lab/presentation_report/report_cards.md",
            source_type="project",
            year="2026",
        ),
        SourceRecord(
            id="project-methods",
            title="Методические материалы по группам методов",
            url="docs/METHOD_GROUPS.md",
            source_type="project",
            year="2026",
        ),
    ]
