"""Persistent state for the local thesis-writing workflow."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diploma_agent import config


@dataclass
class Section:
    id: str
    title: str
    status: str = "planned"
    file: str = ""
    notes: str = ""
    sources: list[str] = field(default_factory=list)
    updated_at: str = ""
    active: bool = True


@dataclass
class ThesisState:
    topic: str = "Разработка многоагентной системы навигации и координации наземных роботов в симуляционной среде"
    plan_markdown: str = ""
    sections: dict[str, Section] = field(default_factory=dict)
    archived_sections: dict[str, Section] = field(default_factory=dict)
    chat: list[dict[str, str]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


DEFAULT_SECTION_RE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)
INTRO_TITLES = ("введение",)
CONCLUSION_TITLES = ("заключение", "выводы")
ANNOTATION_TITLES = ("аннотация",)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_workspace(workspace: Path | None = None) -> Path:
    root = workspace or config.workspace_dir()
    for child in (
        "sections",
        "sources",
        "sources/pdf",
        "sources/web_cache",
        "indexes",
        "build",
        "assets",
        "runs",
        "quality/reports",
        "versions",
    ):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def state_path(workspace: Path | None = None) -> Path:
    return ensure_workspace(workspace) / "state.json"


def section_filename(section_id: str, title: str) -> str:
    slug = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", title, flags=re.UNICODE).strip("_")
    return f"{section_id.replace('.', '_')}_{slug[:48] or 'section'}.md"


def _section_id_from_title(title: str, fallback: int) -> str | None:
    clean = title.strip()
    lower = clean.lower()
    if any(lower.startswith(item) for item in INTRO_TITLES):
        return "intro"
    if any(lower.startswith(item) for item in CONCLUSION_TITLES):
        return "conclusion"
    if any(lower.startswith(item) for item in ANNOTATION_TITLES):
        return "annotation"
    chapter_match = re.search(r"\bглава\s+(\d+)(?:\D|$)", clean, re.IGNORECASE)
    if chapter_match:
        return chapter_match.group(1)
    number_match = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", clean)
    if number_match:
        return number_match.group(1)
    if lower.startswith("что вставить") or lower.startswith("титульные"):
        return None
    return None if fallback > 1 else str(fallback)


def parse_sections(plan_markdown: str) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    fallback_counter = 1
    for match in DEFAULT_SECTION_RE.finditer(plan_markdown):
        title = match.group(2).strip()
        section_id = _section_id_from_title(title, fallback_counter)
        fallback_counter += 1
        if section_id is None:
            continue
        sections[section_id] = Section(
            id=section_id,
            title=title,
            file=section_filename(section_id, title),
            updated_at=now_iso(),
        )
    return sections


def default_plan() -> str:
    draft = config.ROOT / "docs" / "DIPLOMA_CHAPTER_DRAFT.md"
    if draft.exists():
        return draft.read_text(encoding="utf-8")
    return "# План диплома\n\n## Глава 1. Постановка задачи\n\n## Глава 2. Обзор методов\n"


def initial_state() -> ThesisState:
    plan = default_plan()
    return ThesisState(
        plan_markdown=plan,
        sections=parse_sections(plan),
        settings={
            "workspace": str(config.workspace_dir()),
            "created_at": now_iso(),
            "model": config.diploma_model(),
        },
    )


def load_state(workspace: Path | None = None) -> ThesisState:
    path = state_path(workspace)
    if not path.exists():
        state = initial_state()
        save_state(state, workspace)
        return state
    raw = json.loads(path.read_text(encoding="utf-8"))
    sections = {
        key: _section_from_raw(value) if isinstance(value, dict) else value
        for key, value in raw.get("sections", {}).items()
    }
    archived_sections = {
        key: _section_from_raw(value) if isinstance(value, dict) else value
        for key, value in raw.get("archived_sections", {}).items()
    }
    return ThesisState(
        topic=raw.get("topic", ThesisState().topic),
        plan_markdown=raw.get("plan_markdown", ""),
        sections=sections,
        archived_sections=archived_sections,
        chat=raw.get("chat", []),
        settings=raw.get("settings", {}),
    )


def save_state(state: ThesisState, workspace: Path | None = None) -> None:
    path = state_path(workspace)
    payload = asdict(state)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def section_path(section: Section, workspace: Path | None = None) -> Path:
    root = ensure_workspace(workspace)
    return root / "sections" / section.file


def read_section(section: Section, workspace: Path | None = None) -> str:
    path = section_path(section, workspace)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_section(section: Section, content: str, workspace: Path | None = None) -> None:
    path = section_path(section, workspace)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    section.updated_at = now_iso()
    section.status = "draft"


def merge_sections_with_existing(
    parsed: dict[str, Section],
    existing: dict[str, Section],
    archived: dict[str, Section] | None = None,
) -> tuple[dict[str, Section], dict[str, Section]]:
    archived = dict(archived or {})
    merged: dict[str, Section] = {}
    for section_id, section in parsed.items():
        old = existing.get(section_id) or archived.pop(section_id, None)
        if old:
            old.title = section.title
            old.active = True
            old.updated_at = now_iso()
            merged[section_id] = old
        else:
            merged[section_id] = section
    for section_id, old in existing.items():
        if section_id not in merged:
            old.active = False
            old.updated_at = now_iso()
            archived[section_id] = old
    return merged, archived


def _section_from_raw(raw: dict[str, Any]) -> Section:
    allowed = {item.name for item in fields(Section)}
    clean = {key: value for key, value in raw.items() if key in allowed}
    return Section(**clean)
