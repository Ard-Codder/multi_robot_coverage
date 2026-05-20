"""Streamlit panel for the local thesis agent.

Run:
    python -m streamlit run diploma_agent/app.py
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from diploma_agent import config
from diploma_agent.antiplag.providers import ImportOnlyProvider
from diploma_agent.llm_client import LMStudioClient
from diploma_agent.orchestrator import DiplomaOrchestrator
from diploma_agent.retrievers.pdf_rag import PdfRagIndex
from diploma_agent.state import Section, load_state, read_section, save_state, section_path, write_section
from diploma_agent.versioning import list_versions, workspace_stats


def _orchestrator() -> DiplomaOrchestrator:
    return DiplomaOrchestrator()


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


def _section_options(state) -> list[str]:
    return [
        f"{sid} — {section.title} [{section.status}]"
        for sid, section in sorted(state.sections.items(), key=_section_sort_key)
    ]


def _selected_id(label: str) -> str:
    return label.split(" — ", 1)[0]


def _safe_asset_name(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", stem, flags=re.UNICODE).strip("_") or "image"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{stem[:48]}{suffix}"


def _save_image(uploaded_file, caption: str) -> str:
    assets = config.workspace_dir() / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    filename = _safe_asset_name(uploaded_file.name)
    target = assets / filename
    target.write_bytes(uploaded_file.getbuffer())
    return f"![{caption or Path(uploaded_file.name).stem}](../assets/{filename})"


def _show_result(result) -> None:
    st.success(result.message)
    if result.content:
        st.markdown(result.content)
    if result.sources:
        with st.expander("Контекст, который получила модель"):
            for source in result.sources:
                st.markdown(f"- `{source.source}`: {source.title} ({source.score:.2f})")


def main() -> None:
    st.set_page_config(page_title="Local Thesis Agent", layout="wide")
    st.title("Локальный дипломный агент")
    st.caption("Редактор диплома: план → глава → ревью → Typst/PDF. Токены тратит локальная Gemma в LM Studio.")

    orch = _orchestrator()
    state = load_state(config.workspace_dir())

    with st.sidebar:
        st.subheader("LM Studio")
        st.code(config.lmstudio_base_url())
        st.text_input("Model", value=config.diploma_model(), key="model_hint", disabled=True)
        if st.button("Проверить модель"):
            ok, message = LMStudioClient().is_available()
            (st.success if ok else st.error)(message)

        st.subheader("Workspace")
        st.code(str(config.workspace_dir()))
        st.caption("Эта папка исключена из git через `.gitignore`.")
        st.metric("Глав в плане", len(state.sections))
        st.metric("Архивных глав", len(state.archived_sections))
        stats = workspace_stats(config.workspace_dir())
        st.metric("Текущий объём, стр.", stats["approx_pages"])

        st.subheader("PDF источники")
        uploaded = st.file_uploader("Добавить PDF в приватный RAG", type=["pdf"], accept_multiple_files=True)
        if uploaded and st.button("Индексировать PDF"):
            index = PdfRagIndex(config.workspace_dir())
            added = 0
            for file in uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    source_copy = config.workspace_dir() / "sources" / file.name
                    source_copy.write_bytes(tmp_path.read_bytes())
                    added += index.add_pdf(source_copy)
                finally:
                    tmp_path.unlink(missing_ok=True)
            st.success(f"Добавлено PDF chunks: {added}")

    plan_tab, full_run_tab, editor_tab, export_tab = st.tabs(["План", "Полный прогон", "Редактор глав", "Экспорт и индексы"])

    with plan_tab:
        st.subheader("Режим плана")
        st.write("Сгенерируй структуру, поправь markdown вручную и сохрани. Уже написанные главы не удаляются, а уходят в архив.")
        note = st.text_input(
            "Комментарий к генерации плана",
            value="Короткий план: 5 глав, по 2-4 подраздела, строго по coverage_lab/experiments_lab/results.",
        )
        plan_actions = st.columns([1, 1, 3])
        if plan_actions[0].button("Придумать план", type="primary"):
            try:
                _show_result(orch.generate_plan(note))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        if plan_actions[1].button("Сохранить план"):
            try:
                _show_result(orch.update_plan(st.session_state.get("plan_editor", state.plan_markdown)))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        st.text_area("Markdown-план", value=state.plan_markdown, height=560, key="plan_editor")

        st.subheader("Распознанные главы")
        for sid, section in sorted(state.sections.items(), key=_section_sort_key):
            st.markdown(f"- `{sid}` **{section.title}** — `{section.status}` → `{section.file}`")
        if state.archived_sections:
            with st.expander("Архивные главы, сохраненные после изменения плана"):
                for sid, section in sorted(state.archived_sections.items(), key=_section_sort_key):
                    st.markdown(f"- `{sid}` **{section.title}** → `{section.file}`")

    with full_run_tab:
        st.subheader("Полный прогон")
        st.write("Одна кнопка проходит по плану, пишет главы, показывает прогресс, сохраняет отчеты качества и собирает DOCX/PDF.")
        mode = st.selectbox(
            "Режим генерации",
            [
                ("short", "Короткий черновик"),
                ("normal", "Нормальный черновик"),
                ("outline", "Тезисный черновик"),
                ("target_pages_80", "Целевой прогон 80 страниц"),
            ],
            format_func=lambda item: item[1],
        )[0]
        target_pages = st.number_input("Целевой объём страниц", min_value=10, max_value=120, value=80, step=5)
        budget_result = orch.page_budget(int(target_pages))
        with st.expander("Бюджет по главам"):
            st.markdown(budget_result.content)
        opts = st.columns(4)
        include_plan = opts[0].checkbox("Сначала обновить план", value=False)
        only_empty = opts[1].checkbox("Писать только пустые главы", value=True)
        regenerate = opts[2].checkbox("Перегенерировать всё", value=False)
        save_snapshot = opts[3].checkbox("Сохранить версию после прогона", value=True)
        snapshot_label = st.text_input("Название версии", value=f"draft_{int(target_pages)}_pages_v1")
        if st.button("Сгенерировать план и главы", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
            events = []
            for event in orch.iter_full_pipeline(
                mode=mode,
                regenerate=regenerate,
                include_plan=include_plan,
                only_empty=only_empty,
                target_pages=int(target_pages),
                snapshot_label=snapshot_label,
                save_snapshot=save_snapshot,
            ):
                events.append(event)
                status.info(f"{event.step}: {event.section_id} {event.title} — {event.detail or event.status}")
                if event.preview:
                    preview.markdown(event.preview[:1200])
                done = len([item for item in events if item.status in {"done", "skipped"}])
                total = max(len(state.sections), 1)
                progress.progress(min(done / total, 1.0))
            st.success("Полный прогон завершен.")
        st.caption("Журналы прогонов сохраняются в `thesis_workspace/runs/`.")

        st.subheader("История версий")
        versions = list_versions(config.workspace_dir())
        if versions:
            for item in versions[-8:][::-1]:
                st.markdown(f"- `{item.id}` — {item.label}, ~{item.approx_pages} стр., {item.chars} зн.")
            vc = st.columns(3)
            if vc[0].button("Сравнить две последние"):
                _show_result(orch.diff_versions())
            if vc[1].button("Сохранить текущую версию"):
                _show_result(orch.snapshot(label=snapshot_label, reason="manual ui snapshot", target_pages=int(target_pages)))
            restore_id = vc[2].text_input("Restore id")
            if restore_id and st.button("Восстановить версию"):
                _show_result(orch.restore_version(restore_id))
            prune_keep = st.text_input("Оставить версии с меткой", value="final_clean_export")
            if st.button("Удалить промежуточные версии"):
                _show_result(orch.prune_versions(keep_label=prune_keep))
        else:
            st.info("Сохранённых версий пока нет.")

    with editor_tab:
        st.subheader("Редактор глав")
        options = _section_options(state)
        if not options:
            st.warning("План пока не содержит распознанных разделов.")
            return
        selected = st.selectbox("Глава / раздел", options)
        section_id = _selected_id(selected)
        section = state.sections[section_id]
        current = read_section(section, config.workspace_dir())
        st.caption(f"Файл: `{section_path(section, config.workspace_dir())}`")

        gen_note = st.text_input(
            "Задание для генерации/перегенерации",
            value="Пиши академично, связно, без воды. Если данных мало, пометь [ТРЕБУЕТ УТОЧНЕНИЯ].",
        )
        action_cols = st.columns([1, 1, 1, 1, 2])
        if action_cols[0].button("Написать", type="primary"):
            try:
                result = orch.write_section(section_id, gen_note)
                _show_result(result)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        if action_cols[1].button("Перегенерить"):
            try:
                result = orch.rewrite_section(section_id, gen_note or "Перепиши раздел лучше и связнее.")
                _show_result(result)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        if action_cols[2].button("Review"):
            try:
                result = orch.review_section(section_id, gen_note)
                _show_result(result)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        if action_cols[3].button("Источники"):
            _show_result(orch.list_sources(section_id))
        if st.button("Локальный анализ качества"):
            _show_result(orch.analyze_section_quality(section_id))

        image_cols = st.columns([2, 2, 1])
        image = image_cols[0].file_uploader("Картинка в раздел", type=["png", "jpg", "jpeg", "webp"], key=f"img_{section_id}")
        caption = image_cols[1].text_input("Подпись", key=f"caption_{section_id}")
        if image and image_cols[2].button("Вставить", key=f"insert_{section_id}"):
            snippet = _save_image(image, caption)
            st.session_state[f"section_editor_{section_id}"] = (
                st.session_state.get(f"section_editor_{section_id}", current).rstrip() + "\n\n" + snippet + "\n"
            )
            st.rerun()

        edited = st.text_area(
            "Текст раздела",
            value=current,
            height=620,
            key=f"section_editor_{section_id}",
        )
        if st.button("Сохранить текст вручную"):
            write_section(section, edited, config.workspace_dir())
            save_state(state, config.workspace_dir())
            st.success("Раздел сохранен.")

        if section.sources:
            with st.expander("Источники раздела"):
                for source in section.sources:
                    st.markdown(f"- {source}")

    with export_tab:
        st.subheader("Экспорт и служебные операции")
        cols = st.columns(4)
        if cols[0].button("DOCX для Word", type="primary"):
            result = orch.render_docx()
            st.success(result.message)
            st.code(result.content)
        if cols[1].button("Собрать Typst/PDF"):
            result = orch.render_pdf()
            st.info(result.message)
            st.code(result.content)
        if cols[2].button("Построить code embeddings"):
            try:
                result = orch.build_code_embeddings()
                st.success(result.message)
                st.code(result.content)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        command = cols[3].text_input("Команда", placeholder="/write 2.1")
        if command and st.button("Выполнить команду"):
            try:
                _show_result(orch.handle(command))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        st.subheader("Версии и объём")
        if st.button("Показать версии"):
            _show_result(orch.versions())
        if st.button("Бюджет 80 страниц"):
            _show_result(orch.page_budget(80))

        st.subheader("ГОСТ Typst")
        gost_cols = st.columns(3)
        if gost_cols[0].button("ГОСТ PDF", type="primary"):
            result = orch.render_gost_pdf()
            st.info(result.message)
            st.code(result.content)
        if gost_cols[1].button("ГОСТ Typst"):
            result = orch.render_gost_typst()
            st.info(result.message)
            st.code(result.content)
        if gost_cols[2].button("Проверить ГОСТ"):
            result = orch.validate_gost()
            st.info(result.message)
            st.code(result.content)

        st.subheader("Внешние отчеты уникальности")
        provider = st.selectbox("Сервис", ["SeoLik", "PlagiatAI", "Проверить-уникальность.рф", "Другое"])
        report_url = st.text_input("Ссылка на отчет")
        metrics_cols = st.columns(3)
        originality = metrics_cols[0].number_input("Уникальность, %", min_value=0.0, max_value=100.0, value=0.0)
        ai_percent = metrics_cols[1].number_input("ИИ, %", min_value=0.0, max_value=100.0, value=0.0)
        notes = metrics_cols[2].text_input("Комментарий")
        if st.button("Импортировать отчет"):
            path = ImportOnlyProvider(provider, config.workspace_dir()).import_report(
                report_url,
                originality_percent=originality or None,
                ai_percent=ai_percent or None,
                notes=notes,
            )
            st.success(f"Отчет сохранен: {path}")


if __name__ == "__main__":
    main()
