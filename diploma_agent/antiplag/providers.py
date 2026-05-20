"""API-only antiplagiarism integration points.

No scraping, captcha bypassing or hidden browser automation is implemented here.
If a provider exposes an official API, a concrete subclass can call it. For
services without API access, store manually obtained report links/metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from diploma_agent.sources.source_store import SourceStore


@dataclass(frozen=True)
class AntiplagiarismResult:
    provider: str
    originality_percent: float | None = None
    ai_percent: float | None = None
    report_url: str = ""
    notes: str = ""


class AntiplagiarismProvider(Protocol):
    name: str

    def check_text(self, text: str) -> AntiplagiarismResult:
        """Run a check through an official API."""


class ImportOnlyProvider:
    def __init__(self, name: str, workspace: Path | None = None) -> None:
        self.name = name
        self.store = SourceStore(workspace)

    def import_report(
        self,
        report_url: str,
        originality_percent: float | None = None,
        ai_percent: float | None = None,
        notes: str = "",
    ) -> Path:
        metrics = {
            "originality_percent": originality_percent,
            "ai_percent": ai_percent,
            "notes": notes,
        }
        return self.store.add_external_report(self.name, report_url, metrics)

    def check_text(self, text: str) -> AntiplagiarismResult:
        raise RuntimeError(
            f"{self.name} is import-only until an official API key/endpoint is configured. "
            "Paste report URL/metrics instead of automating the website."
        )
