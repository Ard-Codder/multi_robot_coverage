"""Command orchestrator for the local thesis agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.bibliography.gost import save_bibliography
from diploma_agent.formulas import FORMULA_SNIPPETS, recommended_formula_block
from diploma_agent.llm_client import LLMRequest, LMStudioClient, system_user
from diploma_agent.page_budget import PageBudget, budget_note, build_page_budget
from diploma_agent.quality.analyzer import analyze_text, save_quality_report
from diploma_agent.retrievers.code_summary import CodeSummaryRetriever
from diploma_agent.retrievers.embedding_index import JsonlEmbeddingIndex
from diploma_agent.retrievers.local_docs import LocalDocsRetriever, SourceChunk
from diploma_agent.retrievers.pdf_rag import PdfRagIndex
from diploma_agent.retrievers.web import search_as_chunks
from diploma_agent.renderer.docx import render_docx
from diploma_agent.renderer.typst import render
from diploma_agent.renderer.typst_gost import render_gost, validate_gost_export
from diploma_agent.sources.source_store import SourceStore
from diploma_agent.state import (
    Section,
    ThesisState,
    load_state,
    merge_sections_with_existing,
    parse_sections,
    read_section,
    save_state,
    section_path,
    write_section,
    now_iso,
)
from diploma_agent.versioning import (
    compare_versions,
    create_snapshot,
    list_versions,
    prune_versions,
    restore_version,
    workspace_stats,
)


@dataclass(frozen=True)
class CommandResult:
    message: str
    content: str = ""
    sources: list[SourceChunk] | None = None


@dataclass(frozen=True)
class FullRunEvent:
    step: str
    section_id: str = ""
    title: str = ""
    status: str = "ok"
    preview: str = ""
    detail: str = ""


def _prompt(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def _format_context(chunks: list[SourceChunk], limit_chars: int = 11000) -> str:
    blocks: list[str] = []
    total = 0
    for chunk in chunks:
        block = chunk.as_prompt_block()
        if total + len(block) > limit_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def _fallback_plan() -> str:
    return """# План диплома

## Введение
### Актуальность задачи multi-robot coverage
### Цель, объект, предмет и задачи работы

## Глава 1. Постановка задачи и критерии оценки
### 1.1. Формализация задачи покрытия территории группой наземных роботов
### 1.2. Ограничения симуляционной постановки и принятые допущения
### 1.3. Метрики качества: coverage, distance, efficiency, load balance и safety

## Глава 2. Обзор и прикладная классификация методов
### 2.1. Классические coverage planners: grid, frontier, Voronoi, STC, DARP
### 2.2. MAPF-подходы как слой координации и предотвращения конфликтов
### 2.3. Learning-based методы: PPO и ML-guided hybrid
### 2.4. VLA-модели как перспективный high-level слой, а не реализованный planner

## Глава 3. Архитектура реализованной системы
### 3.1. Структура `coverage_lab` и единый симуляционный цикл
### 3.2. Интерфейс алгоритмов, карта, роботы, препятствия и пешеходы
### 3.3. Batch-запуски в `experiments_lab` и контракт результатов JSON/CSV

## Глава 4. Экспериментальные исследования
### 4.1. Статическая сцена `static_A_long`
### 4.2. Динамическая сцена `dynamic_B_long` с пешеходами
### 4.3. Сравнение методов по покрытию, длине траектории, балансу нагрузки и безопасности

## Глава 5. Анализ результатов и перспективы развития
### 5.1. Сильные и слабые стороны classical, MAPF, RL и ML-guided подходов
### 5.2. Ограничения экспериментов и направления дальнейшей валидации
### 5.3. Возможная роль VLA/LLM как селектора стратегии поверх проверяемых planners

## Заключение
### Основные результаты работы
### Возможные направления дальнейшего развития
"""


def _fallback_section_text(section: Section, chunks: list[SourceChunk]) -> str:
    return f"""## {section.title}

Раздел раскрывает часть выпускной квалификационной работы, связанную с разработкой многоагентной системы навигации и координации наземных роботов в симуляционной среде. Основной акцент сделан на реализованной части проекта: `coverage_lab`, `experiments_lab`, сохраненных результатах в `results/lab` и метриках качества покрытия.

В тексте необходимо связать постановку задачи, реализованные программные модули и результаты экспериментов. Для этого используются материалы проекта, описания алгоритмов, экспериментальные CSV/JSON-результаты и визуальные панели сравнения методов.

Дальнейшее наполнение раздела должно уточнять конкретные алгоритмы, параметры сцен и метрики оценки качества без выхода за пределы реализованной экспериментальной платформы.
"""


class DiplomaOrchestrator:
    def __init__(
        self,
        workspace: Path | None = None,
        llm: LMStudioClient | None = None,
        docs: LocalDocsRetriever | None = None,
        code: CodeSummaryRetriever | None = None,
        pdf: PdfRagIndex | None = None,
    ) -> None:
        self.workspace = workspace or config.workspace_dir()
        self.llm = llm or LMStudioClient()
        self.docs = docs or LocalDocsRetriever()
        self.code = code or CodeSummaryRetriever()
        self.pdf = pdf or PdfRagIndex(self.workspace)
        self.code_embedding_index = JsonlEmbeddingIndex("code", self.workspace)
        self.source_store = SourceStore(self.workspace)

    def load(self) -> ThesisState:
        return load_state(self.workspace)

    def save(self, state: ThesisState) -> None:
        save_state(state, self.workspace)

    def context_pack(self, section: Section, extra_query: str = "", include_web: bool = False) -> list[SourceChunk]:
        query = f"{section.id} {section.title} {extra_query}".strip()
        chunks: list[SourceChunk] = []
        chunks.extend(self.docs.search(query, top_k=7))
        chunks.extend(self.code.search(query, top_k=5))
        if self.code_embedding_index.path.exists():
            try:
                chunks.extend(self.code_embedding_index.search(self.llm.embed([query])[0], top_k=5))
            except RuntimeError:
                pass
        chunks.extend(self.pdf.search(query, top_k=4))
        if include_web:
            chunks.extend(search_as_chunks(query, max_results=4))
        for chunk in chunks[:8]:
            if chunk.source.startswith(("http://", "https://")):
                self.source_store.add_chunk(chunk, section.id)
        return chunks

    def generate_plan(self, user_note: str = "") -> CommandResult:
        state = self.load()
        chunks = self.docs.search("план диплома структура главы цель задачи эксперимент результаты", top_k=10)
        user = (
            f"Тема диплома:\n{state.topic}\n\n"
            f"Текущий план/черновик:\n{state.plan_markdown[:6000]}\n\n"
            f"Комментарий пользователя:\n{user_note or '(нет)'}\n\n"
            f"Контекст проекта:\n{_format_context(chunks)}"
        )
        try:
            plan = self.llm.chat(LLMRequest(system_user(_prompt("planner"), user), role="planner", max_tokens=8192))
        except RuntimeError:
            plan = _fallback_plan()
        if not plan.strip():
            plan = _fallback_plan()
        lowered = plan.lower()
        if "ros/gazebo" in lowered or "matlab" in lowered:
            plan = _fallback_plan()
        state.plan_markdown = plan
        parsed = parse_sections(plan)
        state.sections, state.archived_sections = merge_sections_with_existing(
            parsed,
            state.sections,
            state.archived_sections,
        )
        self.save(state)
        return CommandResult("План обновлен.", plan, chunks)

    def update_plan(self, new_plan: str) -> CommandResult:
        state = self.load()
        state.plan_markdown = new_plan.strip()
        parsed = parse_sections(state.plan_markdown)
        state.sections, state.archived_sections = merge_sections_with_existing(
            parsed,
            state.sections,
            state.archived_sections,
        )
        self.save(state)
        return CommandResult("План сохранен.", state.plan_markdown)

    def write_section(self, section_id: str, user_note: str = "", include_web: bool = False) -> CommandResult:
        state = self.load()
        section = self._section_or_raise(state, section_id)
        chunks = self.context_pack(section, user_note, include_web=include_web)
        user = (
            f"Тема диплома:\n{state.topic}\n\n"
            f"План диплома:\n{state.plan_markdown[:6000]}\n\n"
            f"Нужно написать раздел {section.id}: {section.title}\n"
            f"Комментарий пользователя:\n{user_note or '(нет)'}\n\n"
            f"Контекст:\n{_format_context(chunks)}"
        )
        try:
            text = self.llm.chat(LLMRequest(system_user(_prompt("writer"), user), role="writer", max_tokens=8192))
        except RuntimeError:
            text = _fallback_section_text(section, chunks)
        text = _postprocess_section_text(section, text)
        section.sources = [f"{chunk.source} :: {chunk.title}" for chunk in chunks]
        write_section(section, text, self.workspace)
        save_quality_report(text, f"section_{section.id.replace('.', '_')}", self.workspace)
        self.save(state)
        return CommandResult(f"Раздел {section.id} сохранен: {section_path(section, self.workspace)}", text, chunks)

    def write_section_with_budget(
        self,
        section_id: str,
        budget: PageBudget,
        user_note: str = "",
        include_web: bool = False,
    ) -> CommandResult:
        note = "\n".join(part for part in (user_note, budget_note(section_id, budget)) if part)
        result = self.write_section(section_id, note, include_web)
        section_budget = budget.sections.get(section_id)
        if section_budget and len(result.content) < int(section_budget.target_chars * 0.65):
            expand_note = (
                f"Расширь этот раздел до уровня бакалаврской ВКР: примерно {section_budget.target_words} слов "
                f"и {section_budget.target_chars} знаков. Не добавляй воду: добавь конкретику проекта, "
                "описание алгоритмов/метрик/экспериментов, связные переходы и пояснения. "
                "Сохрани уже написанный смысл и заверши текст полным предложением."
            )
            result = self.rewrite_section(section_id, expand_note, include_web)
        return result

    def rewrite_section(self, section_id: str, user_note: str, include_web: bool = False) -> CommandResult:
        state = self.load()
        section = self._section_or_raise(state, section_id)
        current = read_section(section, self.workspace)
        chunks = self.context_pack(section, user_note, include_web=include_web)
        user = (
            f"Раздел {section.id}: {section.title}\n\n"
            f"Текущий текст:\n{current}\n\n"
            f"Как переписать:\n{user_note}\n\n"
            f"Контекст:\n{_format_context(chunks)}"
        )
        try:
            text = self.llm.chat(LLMRequest(system_user(_prompt("writer"), user), role="writer", max_tokens=8192))
        except RuntimeError:
            text = _fallback_section_text(section, chunks)
        text = _postprocess_section_text(section, text)
        section.sources = [f"{chunk.source} :: {chunk.title}" for chunk in chunks]
        write_section(section, text, self.workspace)
        save_quality_report(text, f"section_{section.id.replace('.', '_')}", self.workspace)
        self.save(state)
        return CommandResult(f"Раздел {section.id} переписан.", text, chunks)

    def review_section(self, section_id: str, user_note: str = "") -> CommandResult:
        state = self.load()
        section = self._section_or_raise(state, section_id)
        current = read_section(section, self.workspace)
        chunks = self.context_pack(section, user_note)
        user = (
            f"План диплома:\n{state.plan_markdown[:5000]}\n\n"
            f"Раздел {section.id}: {section.title}\n\n"
            f"Текст раздела:\n{current}\n\n"
            f"Комментарий пользователя:\n{user_note or '(нет)'}\n\n"
            f"Контекст для проверки:\n{_format_context(chunks)}"
        )
        try:
            text = self.llm.chat(LLMRequest(system_user(_prompt("reviewer"), user), role="reviewer", max_tokens=8192))
        except RuntimeError:
            text = current or _fallback_section_text(section, chunks)
        section.status = "reviewed"
        write_section(section, text, self.workspace)
        save_quality_report(text, f"section_{section.id.replace('.', '_')}", self.workspace)
        self.save(state)
        return CommandResult(f"Раздел {section.id} отредактирован reviewer-ом.", text, chunks)

    def analyze_section_quality(self, section_id: str) -> CommandResult:
        state = self.load()
        section = self._section_or_raise(state, section_id)
        text = read_section(section, self.workspace)
        report = analyze_text(text)
        path = save_quality_report(text, f"section_{section.id.replace('.', '_')}", self.workspace)
        return CommandResult("Локальный отчет качества готов.", report.to_markdown() + f"\n\nJSON: `{path}`")

    def list_sources(self, section_id: str) -> CommandResult:
        state = self.load()
        section = self._section_or_raise(state, section_id)
        sources = "\n".join(f"- {source}" for source in section.sources) or "Источники для раздела пока не сохранены."
        return CommandResult("Источники раздела.", sources)

    def render_pdf(self) -> CommandResult:
        save_bibliography(self.workspace)
        result = render(self.load(), self.workspace)
        content = f"Typst: {result.typ_path}\nPDF: {result.pdf_path}\ncompiled={result.compiled}"
        return CommandResult(result.message, content)

    def render_docx(self) -> CommandResult:
        save_bibliography(self.workspace)
        result = render_docx(self.load(), self.workspace)
        return CommandResult(result.message, f"DOCX: {result.docx_path}")

    def render_gost_pdf(self) -> CommandResult:
        result = render_gost(self.load(), self.workspace)
        content = f"Typst: {result.main_typ}\nBibTeX: {result.bib_path}\nPDF: {result.pdf_path}\ncompiled={result.compiled}"
        return CommandResult(result.message, content)

    def render_gost_typst(self) -> CommandResult:
        result = render_gost(self.load(), self.workspace, compile_pdf=False)
        content = f"Typst: {result.main_typ}\nBibTeX: {result.bib_path}\nPDF: {result.pdf_path}\ncompiled={result.compiled}"
        return CommandResult(result.message, content)

    def validate_gost(self) -> CommandResult:
        result = render_gost(self.load(), self.workspace, compile_pdf=False)
        errors = validate_gost_export(result.main_typ)
        if errors:
            return CommandResult("ГОСТ-валидация нашла проблемы.", "\n".join(f"- {e}" for e in errors))
        return CommandResult("ГОСТ-валидация пройдена.", f"Checked: {result.main_typ}")

    def save_bibliography(self) -> CommandResult:
        path = save_bibliography(self.workspace)
        return CommandResult("Список литературы обновлен.", f"Bibliography: {path}")

    def page_budget(self, target_pages: int = 80) -> CommandResult:
        budget = build_page_budget(self.load().sections, target_pages)
        return CommandResult("Бюджет объема рассчитан.", budget.to_markdown())

    def iter_full_pipeline(
        self,
        mode: str = "short",
        regenerate: bool = False,
        include_plan: bool = False,
        only_empty: bool = True,
        target_pages: int = 80,
        snapshot_label: str = "",
        save_snapshot: bool = False,
    ):
        run_log: list[dict] = []
        run_path = self.workspace / "runs" / f"run_{now_iso().replace(':', '-').replace('+', '_')}.json"
        if include_plan:
            yield FullRunEvent("plan", detail="Генерирую план")
            result = self.generate_plan("Бакалаврский план: конкретно по проекту, без диссертационного тона.")
            run_log.append({"step": "plan", "preview": result.content[:1000]})
            yield FullRunEvent("plan", status="done", preview=result.content[:1000])

        state = self.load()
        budget = build_page_budget(state.sections, target_pages)
        run_log.append({"step": "budget", "target_pages": target_pages, "budget": budget.to_markdown()})
        if target_pages >= 50:
            sections = [
                (sid, sec)
                for sid, sec in sorted(state.sections.items(), key=_section_sort_key)
                if sec.active and sid not in {"annotation"}
            ]
        else:
            sections = [
                (sid, sec)
                for sid, sec in sorted(state.sections.items(), key=_section_sort_key)
                if sid in {"intro", "1", "2", "3", "4", "5", "conclusion"} or "." not in sid
            ]
        total = len(sections)
        for idx, (section_id, section) in enumerate(sections, start=1):
            current = read_section(section, self.workspace)
            if only_empty and current.strip() and not regenerate:
                save_quality_report(current, f"section_{section.id.replace('.', '_')}", self.workspace)
                preview = current[:900]
                event = FullRunEvent("skip", section_id, section.title, "skipped", preview, f"{idx}/{total}: уже есть текст")
                run_log.append(event.__dict__)
                yield event
                continue
            yield FullRunEvent("write", section_id, section.title, "running", detail=f"{idx}/{total}: пишу раздел")
            note = _full_run_note(mode)
            result = self.write_section_with_budget(section_id, budget, note, include_web=_section_needs_web(section))
            report = analyze_text(result.content)
            event = FullRunEvent(
                "write",
                section_id,
                section.title,
                "done",
                result.content[:900],
                f"{idx}/{total}: готово, слов={report.words}, среднее предложение={report.avg_sentence_words:.1f}",
            )
            run_log.append(event.__dict__)
            run_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
            yield event

        biblio = save_bibliography(self.workspace)
        docx = self.render_docx()
        pdf = self.render_pdf()
        snapshot_detail = ""
        if save_snapshot:
            snapshot = create_snapshot(
                self.workspace,
                label=snapshot_label or f"draft_{target_pages}_pages",
                reason="full pipeline run",
                target_pages=target_pages,
                run_log=run_log,
            )
            snapshot_detail = f"\nSnapshot: {snapshot.path}"
        final = FullRunEvent("export", status="done", detail=f"{biblio}\n{docx.content}\n{pdf.content}{snapshot_detail}")
        run_log.append(final.__dict__)
        run_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
        yield final

    def snapshot(self, label: str = "", reason: str = "", target_pages: int = 80) -> CommandResult:
        snapshot = create_snapshot(self.workspace, label=label, reason=reason, target_pages=target_pages)
        return CommandResult(
            "Версия сохранена.",
            f"{snapshot.id}\nPath: {snapshot.path}\nPages approx: {snapshot.approx_pages}\nChars: {snapshot.chars}",
        )

    def versions(self) -> CommandResult:
        versions = list_versions(self.workspace)
        if not versions:
            return CommandResult("Версий пока нет.", "")
        lines = ["## Версии"]
        for item in versions[-20:]:
            lines.append(
                f"- `{item.id}` — {item.label}, {item.created_at}, ~{item.approx_pages} стр., {item.chars} зн."
            )
        stats = workspace_stats(self.workspace)
        lines.append("")
        lines.append(f"Текущая рабочая версия: ~{stats['approx_pages']} стр., {stats['chars']} зн., {stats['words']} слов.")
        return CommandResult("Список версий.", "\n".join(lines))

    def diff_versions(self, old_id: str = "previous", new_id: str = "latest") -> CommandResult:
        path = compare_versions(old_id, new_id, self.workspace)
        return CommandResult("Diff готов.", f"Diff: {path}")

    def restore_version(self, version_id: str) -> CommandResult:
        path = restore_version(version_id, self.workspace)
        return CommandResult("Версия восстановлена.", f"Restored: {path}")

    def prune_versions(self, keep_label: str = "final_clean_export", dry_run: bool = False) -> CommandResult:
        removed = prune_versions(self.workspace, keep_labels=[keep_label], keep_latest=True, dry_run=dry_run)
        mode = "Будут удалены" if dry_run else "Удалены"
        content = "\n".join(f"- {item}" for item in removed) or "Промежуточных версий для удаления нет."
        return CommandResult(f"{mode} промежуточные версии.", content)

    def build_code_embeddings(self) -> CommandResult:
        chunks = self.code.chunks()
        texts = [f"{chunk.source}\n{chunk.title}\n{chunk.text}" for chunk in chunks]
        try:
            embeddings = self.llm.embed(texts)
        except RuntimeError as exc:
            return CommandResult(
                "Code embeddings не построены.",
                f"{exc}\n\nЗагрузи embedding-модель в LM Studio или задай DIPLOMA_EMBEDDING_MODEL. "
                "Keyword/code-summary retrieval продолжит работать без embeddings.",
            )
        count = self.code_embedding_index.write(chunks, embeddings)
        return CommandResult("Code embeddings обновлены.", f"Indexed code chunks: {count}")

    def handle(self, command: str) -> CommandResult:
        text = command.strip()
        if not text:
            return CommandResult("Введите команду или комментарий.")
        if text == "/plan":
            return self.generate_plan()
        if text.startswith("/plan "):
            return self.generate_plan(text.removeprefix("/plan ").strip())
        if text.startswith("/update_plan "):
            return self.update_plan(text.removeprefix("/update_plan ").strip())
        if text.startswith("/write "):
            section_id, note = self._split_section_command(text.removeprefix("/write ").strip())
            return self.write_section(section_id, note)
        if text.startswith("/rewrite "):
            section_id, note = self._split_section_command(text.removeprefix("/rewrite ").strip())
            return self.rewrite_section(section_id, note)
        if text.startswith("/review "):
            section_id, note = self._split_section_command(text.removeprefix("/review ").strip())
            return self.review_section(section_id, note)
        if text.startswith("/sources "):
            return self.list_sources(text.removeprefix("/sources ").strip())
        if text.startswith("/quality "):
            return self.analyze_section_quality(text.removeprefix("/quality ").strip())
        if text == "/bibliography":
            return self.save_bibliography()
        if text == "/budget" or text == "/budget_80":
            return self.page_budget(80)
        if text == "/full_run_80":
            events = list(
                self.iter_full_pipeline(
                    mode="target_pages_80",
                    regenerate=False,
                    include_plan=False,
                    only_empty=False,
                    target_pages=80,
                    save_snapshot=True,
                )
            )
            return CommandResult("Полный прогон на 80 страниц завершен.", "\n".join(event.detail for event in events if event.detail))
        if text.startswith("/snapshot"):
            label = text.removeprefix("/snapshot").strip()
            return self.snapshot(label=label, reason="manual snapshot")
        if text == "/versions":
            return self.versions()
        if text.startswith("/diff"):
            parts = text.split()
            old_id = parts[1] if len(parts) > 1 else "previous"
            new_id = parts[2] if len(parts) > 2 else "latest"
            return self.diff_versions(old_id, new_id)
        if text.startswith("/restore "):
            return self.restore_version(text.removeprefix("/restore ").strip())
        if text.startswith("/prune_versions"):
            keep = text.removeprefix("/prune_versions").strip() or "final_clean_export"
            return self.prune_versions(keep_label=keep)
        if text == "/render" or text == "/pdf":
            return self.render_pdf()
        if text == "/docx" or text == "/export_docx":
            return self.render_docx()
        if text == "/gost_pdf":
            return self.render_gost_pdf()
        if text == "/gost_typst":
            return self.render_gost_typst()
        if text == "/gost_validate":
            return self.validate_gost()
        if text == "/index_code":
            return self.build_code_embeddings()
        return CommandResult(
            "Неизвестная команда. Используй /plan, /write 2.1, /quality 2.1, /bibliography, /budget_80, /full_run_80, /snapshot, /versions, /diff, /restore, /render, /docx."
        )

    def _section_or_raise(self, state: ThesisState, section_id: str) -> Section:
        section = state.sections.get(section_id)
        if section is None:
            known = ", ".join(sorted(state.sections)) or "нет разделов"
            raise ValueError(f"Раздел `{section_id}` не найден. Доступные: {known}")
        return section

    @staticmethod
    def _split_section_command(payload: str) -> tuple[str, str]:
        parts = payload.split(maxsplit=1)
        if not parts:
            raise ValueError("Укажи id раздела, например `/write 2.1`.")
        return parts[0], parts[1] if len(parts) > 1 else ""


def _section_sort_key(item: tuple[str, Section]) -> tuple[int, ...]:
    section_id, _ = item
    special = {"annotation": (0,), "intro": (1,), "conclusion": (9998,)}
    if section_id in special:
        return special[section_id]
    parts = []
    for part in section_id.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(9999)
    return tuple(parts)


def _section_needs_web(section: Section) -> bool:
    text = f"{section.id} {section.title}".lower()
    return any(word in text for word in ("обзор", "vla", "метод", "формул", "эксперимент", "literature"))


def _full_run_note(mode: str) -> str:
    size = {
        "short": "2-3 коротких абзаца",
        "normal": "4-6 абзацев",
        "outline": "1-2 абзаца и список тезисов",
        "target_pages_80": "развернутый фрагмент по бюджету главы",
    }.get(mode, "2-3 коротких абзаца")
    return (
        f"Полный прогон: {size}. Стиль бакалаврской ВКР: проще, конкретнее, без канцелярита. "
        "Пиши от лица автора работы: что реализовано в проекте, какие модули используются, какие метрики проверяются. "
        "Метрики, формулы, таблицы и рисунки можно вставлять в теоретические, архитектурные, экспериментальные и аналитические главы, "
        "но не во введение, заключение и список литературы. "
        "Не звучать как кандидатская диссертация. Заверши текст полным предложением."
    )


def _postprocess_section_text(section: Section, text: str) -> str:
    if section.id in {"intro", "conclusion", "annotation"}:
        return text
    additions: list[str] = []
    if "$$" not in text:
        formulas = _formula_block_for_section(section.id)
        if formulas:
            additions.append("## Формулы и метрики\n\n" + formulas)
    if "![" not in text:
        visuals = _visual_block_for_section(section.id)
        if visuals:
            additions.append("## Иллюстрации и результаты\n\n" + visuals)
    if not additions:
        return text
    return text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"


def _formula_block_for_section(section_id: str) -> str:
    if section_id in {"1", "1.3"}:
        return "\n\n".join([FORMULA_SNIPPETS["coverage"], FORMULA_SNIPPETS["load_balance"]])
    if section_id in {"3", "3.3"}:
        return FORMULA_SNIPPETS["efficiency"]
    if section_id in {"4", "4.1", "4.2", "4.3"}:
        return recommended_formula_block()
    if section_id in {"5", "5.1", "5.2"}:
        return "\n\n".join([FORMULA_SNIPPETS["efficiency"], FORMULA_SNIPPETS["pedestrian_clearance"]])
    return ""


def _visual_block_for_section(section_id: str) -> str:
    if section_id in {"4", "4.1", "4.2", "4.3"}:
        return "\n\n".join(
            [
                "![Медианные результаты в статической сцене](../../results/lab/presentation_report/panel_median_static.png)",
                "![Медианные результаты в динамической сцене](../../results/lab/presentation_report/panel_median_dynamic.png)",
            ]
        )
    if section_id in {"5", "5.1", "5.2", "5.3"}:
        return "![Сравнение статической и динамической сцены для seed 0](../../results/lab/presentation_report/panel_seed0_static_dynamic.png)"
    return ""
