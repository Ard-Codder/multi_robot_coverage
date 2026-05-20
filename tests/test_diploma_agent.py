from __future__ import annotations

from pathlib import Path

from diploma_agent.renderer.docx import render_docx
from diploma_agent.renderer.typst import build_typst_document, markdown_to_typst
from diploma_agent.renderer.typst_gost import build_bibtex, build_gost_typ, latex_to_typst_math, render_gost, validate_gost_export
from diploma_agent.retrievers.local_docs import LocalDocsRetriever
from diploma_agent.bibliography.gost import save_bibliography
from diploma_agent.quality.analyzer import analyze_text
from diploma_agent.sources.source_store import SourceStore
from diploma_agent.antiplag.providers import ImportOnlyProvider
from diploma_agent.page_budget import build_page_budget
from diploma_agent.versioning import (
    compare_versions,
    create_snapshot,
    list_versions,
    prune_versions,
    restore_version,
    workspace_stats,
)
from diploma_agent.state import (
    Section,
    ThesisState,
    load_state,
    merge_sections_with_existing,
    parse_sections,
    read_section,
    save_state,
    write_section,
)


def test_state_roundtrip(tmp_path: Path) -> None:
    state = ThesisState(plan_markdown="## Глава 1. Test")
    state.sections["1"] = Section(id="1", title="Глава 1. Test", file="1_test.md")
    save_state(state, tmp_path)

    loaded = load_state(tmp_path)
    assert loaded.sections["1"].title == "Глава 1. Test"

    write_section(loaded.sections["1"], "Текст раздела", tmp_path)
    assert read_section(loaded.sections["1"], tmp_path) == "Текст раздела\n"


def test_local_docs_retriever_finds_tmp_markdown(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("# Coverage\n\nMulti-robot coverage metrics.", encoding="utf-8")

    retriever = LocalDocsRetriever(root=tmp_path, paths=("docs/sample.md",))
    results = retriever.search("coverage metrics", top_k=1)

    assert results
    assert results[0].source == "docs/sample.md"


def test_typst_document_contains_section() -> None:
    state = ThesisState(topic="Test topic")
    state.sections["1"] = Section(id="1", title="Intro", file="intro.md")
    # No section file exists, so renderer should fall back to plan markdown.
    state.plan_markdown = "# План\n\n## Глава 1. Intro"

    doc = build_typst_document(state)

    assert "Test topic" in doc
    assert "= План" in doc


def test_markdown_heading_to_typst() -> None:
    assert markdown_to_typst("## Глава 1\n\nТекст") == "== Глава 1\n\nТекст"


def test_sections_archive_and_restore() -> None:
    old = {"1.1": Section(id="1.1", title="Old", file="old.md", status="draft")}
    merged, archived = merge_sections_with_existing(parse_sections("## Глава 1. New"), old)

    assert "1.1" not in merged
    assert archived["1.1"].file == "old.md"
    assert archived["1.1"].active is False

    merged, archived = merge_sections_with_existing(
        parse_sections("## Глава 1. New\n### 1.1. Restored"),
        merged,
        archived,
    )
    assert merged["1.1"].file == "old.md"
    assert "1.1" not in archived


def test_markdown_image_to_typst_figure() -> None:
    converted = markdown_to_typst("![Схема](../assets/diagram.png)")

    assert '#figure(image("../assets/diagram.png"' in converted
    assert "caption: [Схема]" in converted


def test_docx_renderer_creates_file(tmp_path: Path) -> None:
    state = ThesisState(topic="Test topic")
    state.sections["1"] = Section(id="1", title="Глава 1. Test", file="1_test.md")
    write_section(state.sections["1"], "# Глава 1. Test\n\nТекст раздела.", tmp_path)

    result = render_docx(state, tmp_path)

    assert result.docx_path.exists()
    assert result.docx_path.stat().st_size > 0


def test_quality_analyzer_detects_unfinished_text() -> None:
    report = analyze_text("Это тестовый текст. Последнее предложение оборвано")

    assert report.unfinished is True
    assert report.words > 0


def test_source_store_and_bibliography(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    store.add_external_report("SeoLik", "https://example.com/report", {"originality_percent": 100})

    bibliography = save_bibliography(tmp_path)

    assert bibliography.exists()
    assert "Список использованных источников" in bibliography.read_text(encoding="utf-8")
    assert list((tmp_path / "quality" / "reports").glob("*.json"))


def test_import_only_provider_saves_report(tmp_path: Path) -> None:
    path = ImportOnlyProvider("PlagiatAI", tmp_path).import_report(
        "https://example.com/plagiat",
        originality_percent=52,
        ai_percent=13,
    )

    assert path.exists()


def test_typst_formula_block() -> None:
    converted = markdown_to_typst("$$\ncoverage = |C| / |F|\n$$")

    assert "#align(center)" in converted
    assert "coverage" in converted


def test_page_budget_defaults_to_80_pages() -> None:
    sections = {
        "intro": Section(id="intro", title="Введение", file="intro.md"),
        "1": Section(id="1", title="Глава 1", file="1.md"),
        "4": Section(id="4", title="Глава 4", file="4.md"),
    }

    budget = build_page_budget(sections, 80)

    assert budget.target_pages == 80
    assert budget.sections["4"].target_pages > budget.sections["intro"].target_pages


def test_versioning_snapshot_diff_restore(tmp_path: Path) -> None:
    state = ThesisState(topic="Test topic")
    state.sections["1"] = Section(id="1", title="Глава 1. Test", file="1_test.md")
    write_section(state.sections["1"], "Первая версия", tmp_path)
    first = create_snapshot(tmp_path, label="first", target_pages=80)

    write_section(state.sections["1"], "Вторая версия\n\nНовая строка", tmp_path)
    second = create_snapshot(tmp_path, label="second", target_pages=80)
    diff = compare_versions(first.id, second.id, tmp_path)

    assert first.id != second.id
    assert len(list_versions(tmp_path)) == 2
    assert diff.exists()
    assert workspace_stats(tmp_path)["chars"] > 0
    assert (second.path / "manifest.json").exists()

    restore_version(first.id, tmp_path)
    assert read_section(state.sections["1"], tmp_path) == "Первая версия\n"


def test_prune_versions_keeps_label(tmp_path: Path) -> None:
    state = ThesisState(topic="Test topic")
    state.sections["1"] = Section(id="1", title="Глава 1. Test", file="1_test.md")
    write_section(state.sections["1"], "Первая версия", tmp_path)
    create_snapshot(tmp_path, label="intermediate", target_pages=80)
    create_snapshot(tmp_path, label="final_clean_export", target_pages=80)

    removed = prune_versions(tmp_path, keep_labels=["final_clean_export"], keep_latest=False)

    assert len(removed) == 1
    versions = list_versions(tmp_path)
    assert len(versions) == 1
    assert "final_clean_export" in versions[0].label


def test_gost_math_normalizer() -> None:
    assert latex_to_typst_math(r"\frac{|C|}{|F|}") == "|C| / |F|"
    assert "RR" in latex_to_typst_math(r"\mathbb{R}^2")


def test_gost_renderer_creates_typst_and_bib(tmp_path: Path) -> None:
    state = ThesisState(topic="Test topic")
    state.sections["1"] = Section(id="1", title="Глава 1. Test", file="1_test.md")
    write_section(state.sections["1"], "# Глава 1. Test\n\nТекст раздела.\n\n$$\ncoverage = \\frac{|C|}{|F|}\n$$", tmp_path)

    result = render_gost(state, tmp_path, compile_pdf=False)
    text = result.main_typ.read_text(encoding="utf-8")

    assert result.main_typ.exists()
    assert result.bib_path.exists()
    assert "#show: gost.with" in text
    assert "#outline()" in text
    assert "#bibliography" in text
    assert not validate_gost_export(result.main_typ)
    assert "project-readme" in build_bibtex(tmp_path)
    assert "Глава 1" in build_gost_typ(state, tmp_path, {})
