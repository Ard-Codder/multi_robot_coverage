"""Heuristic quality analysis for bachelor-level thesis text.

This is intentionally local and transparent. It does not try to bypass AI
detectors; it points out overly generic, bureaucratic, repetitive or unfinished
text so the draft can become more natural and project-specific.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from diploma_agent.state import ensure_workspace, now_iso

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]+", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?…]?", re.UNICODE)

STOP_WORDS = {
    "в",
    "и",
    "на",
    "с",
    "по",
    "для",
    "что",
    "как",
    "это",
    "при",
    "также",
    "данной",
    "данный",
    "который",
    "которая",
    "которые",
    "является",
    "являются",
}

CLICHES = (
    "в рамках данной работы",
    "следует отметить",
    "таким образом",
    "актуальность обусловлена",
    "на сегодняшний день",
    "имеет важное значение",
    "позволяет сделать вывод",
)

BACHELOR_REWRITE_HINT = (
    "Пиши проще: меньше канцелярита, больше конкретики проекта. "
    "Сохраняй академичность, но избегай диссертационного тона."
)


@dataclass
class QualityReport:
    chars: int
    words: int
    sentences: int
    avg_sentence_words: float
    stop_word_ratio: float
    repeated_terms: list[tuple[str, int]] = field(default_factory=list)
    cliches: list[str] = field(default_factory=list)
    unfinished: bool = False
    recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Локальный отчет качества",
            f"- Символов: {self.chars}",
            f"- Слов: {self.words}",
            f"- Предложений: {self.sentences}",
            f"- Средняя длина предложения: {self.avg_sentence_words:.1f} слов",
            f"- Доля частых служебных слов: {self.stop_word_ratio:.1%}",
        ]
        if self.repeated_terms:
            lines.append("- Часто повторяются: " + ", ".join(f"{w} ({c})" for w, c in self.repeated_terms[:8]))
        if self.cliches:
            lines.append("- Шаблонные обороты: " + ", ".join(self.cliches))
        if self.unfinished:
            lines.append("- Последнее предложение похоже на обрыв.")
        if self.recommendations:
            lines.append("\n### Рекомендации")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def analyze_text(text: str) -> QualityReport:
    words = [w.lower() for w in WORD_RE.findall(text)]
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    counts: dict[str, int] = {}
    for word in words:
        if len(word) < 5 or word in STOP_WORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    repeated = sorted(((w, c) for w, c in counts.items() if c >= 5), key=lambda item: item[1], reverse=True)
    lower = text.lower()
    cliches = [item for item in CLICHES if item in lower]
    unfinished = bool(text.strip()) and text.strip()[-1] not in ".!?…"
    report = QualityReport(
        chars=len(text),
        words=len(words),
        sentences=len(sentences),
        avg_sentence_words=len(words) / max(len(sentences), 1),
        stop_word_ratio=sum(1 for w in words if w in STOP_WORDS) / max(len(words), 1),
        repeated_terms=repeated[:12],
        cliches=cliches,
        unfinished=unfinished,
    )
    report.recommendations = _recommend(report)
    return report


def save_quality_report(text: str, name: str, workspace: Path | None = None) -> Path:
    root = ensure_workspace(workspace)
    report = analyze_text(text)
    path = root / "quality" / "reports" / f"{name}.json"
    path.write_text(json.dumps({"created_at": now_iso(), **asdict(report)}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _recommend(report: QualityReport) -> list[str]:
    recommendations = [BACHELOR_REWRITE_HINT]
    if report.avg_sentence_words > 24:
        recommendations.append("Разбей длинные предложения: для бакалаврской ВКР лучше 14-22 слова в среднем.")
    if report.stop_word_ratio > 0.28:
        recommendations.append("Сократи вводные слова и связки, добавь больше предметных существительных и фактов проекта.")
    if report.cliches:
        recommendations.append("Замени шаблонные обороты на прямые формулировки: что реализовано, где проверено, какая метрика.")
    if report.unfinished:
        recommendations.append("Дописать последнее предложение: текст не должен заканчиваться обрывом.")
    if report.repeated_terms:
        recommendations.append("Проверь повторы терминов и замени часть повторов местоимениями или уточняющими фразами.")
    return recommendations
