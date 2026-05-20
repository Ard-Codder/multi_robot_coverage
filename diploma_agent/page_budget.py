"""Page and text volume budgeting for long thesis drafts."""

from __future__ import annotations

from dataclasses import dataclass

from diploma_agent.state import Section

CHARS_PER_PAGE_MIN = 1600
CHARS_PER_PAGE_MAX = 2000
CHARS_PER_PAGE_TARGET = 1750

PAGE_WEIGHTS = {
    "intro": 5.5,
    "1": 13.0,
    "2": 17.0,
    "3": 13.0,
    "4": 21.0,
    "5": 9.0,
    "conclusion": 3.5,
}


@dataclass(frozen=True)
class SectionBudget:
    section_id: str
    title: str
    target_pages: float
    target_chars: int
    target_words: int


@dataclass(frozen=True)
class PageBudget:
    target_pages: int
    target_chars: int
    min_chars: int
    max_chars: int
    sections: dict[str, SectionBudget]

    def to_markdown(self) -> str:
        lines = [
            f"## Бюджет объёма: {self.target_pages} страниц",
            f"- Целевой объём: около {self.target_chars:,} знаков".replace(",", " "),
            f"- Допустимый диапазон: {self.min_chars:,}-{self.max_chars:,} знаков".replace(",", " "),
            "",
            "| Раздел | Страницы | Знаки | Слова |",
            "|---|---:|---:|---:|",
        ]
        for item in self.sections.values():
            lines.append(
                f"| {item.section_id} {item.title} | {item.target_pages:.1f} | {item.target_chars} | {item.target_words} |"
            )
        return "\n".join(lines)


def build_page_budget(sections: dict[str, Section], target_pages: int = 80) -> PageBudget:
    top_ids = [sid for sid in PAGE_WEIGHTS if sid in sections]
    if not top_ids:
        top_ids = list(sections)[:1]
    total_weight = sum(PAGE_WEIGHTS.get(sid, 1.0) for sid in top_ids) or 1.0
    text_pages = max(target_pages - 8, 1)  # title, contents, figures and bibliography headroom
    budget_sections: dict[str, SectionBudget] = {}
    for sid in top_ids:
        top_pages = text_pages * PAGE_WEIGHTS.get(sid, 1.0) / total_weight
        children = sorted(child for child in sections if child.startswith(f"{sid}."))
        if children:
            _add_budget(budget_sections, sections[sid], top_pages * 0.30)
            child_pages = top_pages * 0.70 / len(children)
            for child in children:
                _add_budget(budget_sections, sections[child], child_pages)
        else:
            _add_budget(budget_sections, sections[sid], top_pages)
    return PageBudget(
        target_pages=target_pages,
        target_chars=target_pages * CHARS_PER_PAGE_TARGET,
        min_chars=target_pages * CHARS_PER_PAGE_MIN,
        max_chars=target_pages * CHARS_PER_PAGE_MAX,
        sections=budget_sections,
    )


def _add_budget(target: dict[str, SectionBudget], section: Section, pages: float) -> None:
    chars = int(pages * CHARS_PER_PAGE_TARGET)
    target[section.id] = SectionBudget(
        section_id=section.id,
        title=section.title,
        target_pages=pages,
        target_chars=chars,
        target_words=max(int(chars / 7.5), 1),
    )


def budget_note(section_id: str, budget: PageBudget | None) -> str:
    if not budget or section_id not in budget.sections:
        return ""
    item = budget.sections[section_id]
    return (
        f"Целевой объём раздела: примерно {item.target_pages:.1f} стр., "
        f"{item.target_words} слов / {item.target_chars} знаков. "
        "Если не хватает контекста, пиши содержательный черновик и помечай места для расширения."
    )
