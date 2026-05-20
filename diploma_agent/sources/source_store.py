"""Local JSON source store for papers, web pages and project artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from diploma_agent.retrievers.local_docs import SourceChunk
from diploma_agent.state import ensure_workspace, now_iso


@dataclass
class SourceRecord:
    id: str
    title: str
    url: str = ""
    source_type: str = "web"
    authors: list[str] = field(default_factory=list)
    year: str = ""
    container: str = ""
    accessed_at: str = ""
    note: str = ""
    section_ids: list[str] = field(default_factory=list)


class SourceStore:
    def __init__(self, workspace: Path | None = None) -> None:
        self.root = ensure_workspace(workspace)
        self.path = self.root / "sources" / "literature.json"

    def load(self) -> dict[str, SourceRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {key: SourceRecord(**value) for key, value in raw.items()}

    def save(self, records: dict[str, SourceRecord]) -> None:
        self.path.write_text(
            json.dumps({key: asdict(value) for key, value in records.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def upsert(self, record: SourceRecord) -> SourceRecord:
        records = self.load()
        existing = records.get(record.id)
        if existing:
            merged_sections = sorted(set(existing.section_ids) | set(record.section_ids))
            record = SourceRecord(**{**asdict(existing), **asdict(record), "section_ids": merged_sections})
        records[record.id] = record
        self.save(records)
        return record

    def add_chunk(self, chunk: SourceChunk, section_id: str = "") -> SourceRecord:
        url = chunk.source
        source_type = "project" if not chunk.source.startswith(("http://", "https://")) else "web"
        record = SourceRecord(
            id=_stable_id(chunk.source + "::" + chunk.title),
            title=chunk.title or chunk.source,
            url=url,
            source_type=source_type,
            accessed_at=now_iso(),
            note=chunk.text[:500],
            section_ids=[section_id] if section_id else [],
        )
        return self.upsert(record)

    def add_external_report(self, provider: str, url: str, metrics: dict[str, Any]) -> Path:
        reports = self.root / "quality" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        payload = {"provider": provider, "url": url, "metrics": metrics, "created_at": now_iso()}
        path = reports / f"{provider}_{_stable_id(url)[:10]}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:16]
