"""GOST 7.32-2017 export using modern-g7-32."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.renderer.typst import _find_typst
from diploma_agent.sources.source_store import SourceRecord, SourceStore
from diploma_agent.state import ThesisState, ensure_workspace, read_section


@dataclass(frozen=True)
class GostRenderResult:
    main_typ: Path
    bib_path: Path
    pdf_path: Path
    compiled: bool
    message: str


FORBIDDEN_PATTERNS = (
    r"\\frac",
    r"\\mathcal",
    r"\\mathbb",
    r"\\sigma",
    r"\\mu",
    r"\\text",
    r"\[ТРЕБУЕТ УТОЧНЕНИЯ\]",
    r"автоматически подготовленный",
    r"Текст требует ручной",
    r"Ключевые опорные материалы",
)


def render_gost(state: ThesisState, workspace: Path | None = None, compile_pdf: bool = True) -> GostRenderResult:
    root = ensure_workspace(workspace or config.workspace_dir())
    build = root / "build"
    gost_dir = build / "gost"
    images_dir = gost_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    config_path = ensure_gost_config(root, state)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    bib_path = gost_dir / "references.bib"
    bib_path.write_text(build_bibtex(root), encoding="utf-8")

    main_typ = gost_dir / "main.typ"
    main_typ.write_text(build_gost_typ(state, root, cfg), encoding="utf-8")

    pdf_path = build / "diploma_gost.pdf"
    if not compile_pdf:
        return GostRenderResult(main_typ, bib_path, pdf_path, False, "ГОСТ Typst собран, компиляция отключена.")
    typst = _find_typst()
    if not typst:
        return GostRenderResult(main_typ, bib_path, pdf_path, False, "Typst CLI не найден.")
    try:
        subprocess.run(
            [typst, "compile", "--root", str(gost_dir), str(main_typ), str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return GostRenderResult(main_typ, bib_path, pdf_path, False, f"Typst GOST compile failed: {msg}")
    return GostRenderResult(main_typ, bib_path, pdf_path, True, "ГОСТ PDF собран.")


def ensure_gost_config(workspace: Path, state: ThesisState) -> Path:
    path = workspace / "gost_config.json"
    if path.exists():
        return path
    cfg = {
        "ministry": "Министерство науки и высшего образования Российской Федерации",
        "organization": {
            "full": "Московский авиационный институт (национальный исследовательский университет)",
            "short": "МАИ",
        },
        "report_type": "Выпускная квалификационная работа",
        "subject": state.topic,
        "student": {"name": "Кирюшин Артем Максимович", "group": "М3О-412Б-22"},
        "manager": {"name": "Фамилия И.О.", "position": "Руководитель ВКР", "title": "Руководитель"},
        "city": "Москва",
        "year": 2026,
        "udk": "",
    }
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_gost_typ(state: ThesisState, workspace: Path, cfg: dict) -> str:
    body = [
        '#import "@preview/modern-g7-32:0.2.0": abstract, appendixes, enum-numbering, gost',
        "",
        "#set enum(numbering: enum-numbering)",
        "",
        "#show: gost.with(",
        f'  ministry: "{_t(cfg.get("ministry", ""))}",',
        "  organization: (",
        f'    full: "{_t(cfg.get("organization", {}).get("full", ""))}",',
        f'    short: "{_t(cfg.get("organization", {}).get("short", ""))}",',
        "  ),",
        f'  udk: "{_t(cfg.get("udk", ""))}",',
        f'  report-type: "{_t(cfg.get("report_type", "Выпускная квалификационная работа"))}",',
        f'  subject: "{_t(cfg.get("subject", state.topic))}",',
        "  manager: (",
        f'    name: "{_t(cfg.get("manager", {}).get("name", ""))}",',
        f'    position: "{_t(cfg.get("manager", {}).get("position", ""))}",',
        f'    title: "{_t(cfg.get("manager", {}).get("title", "Руководитель"))}",',
        "  ),",
        f'  year: {int(cfg.get("year", 2026))},',
        f'  city: "{_t(cfg.get("city", "Москва"))}",',
        "  text-size: (default: 14pt, small: 10pt),",
        "  indent: 1.25cm,",
        "  pagination-align: center,",
        "  margin: (left: 30mm, right: 15mm, top: 20mm, bottom: 20mm),",
        "  add-pagebreaks: true,",
        "  performers: (",
        "    (",
        f'      name: "{_t(cfg.get("student", {}).get("name", ""))}",',
        f'      position: "студент группы {_t(cfg.get("student", {}).get("group", ""))}",',
        "    ),",
        "  ),",
        ")",
        "",
        '#abstract("multi-robot coverage", "coverage path planning", "MAPF", "RL", "ML-guided")[ ',
        "  В выпускной квалификационной работе рассматривается задача покрытия территории группой автономных наземных роботов.",
        "  Разработанная экспериментальная платформа используется для сопоставления классических, MAPF-, RL- и ML-guided подходов.",
        "]",
        "",
        "#outline()",
        "",
    ]
    seen_images: set[str] = set()
    for section_id in sorted(state.sections, key=_section_sort_key):
        section = state.sections[section_id]
        text = read_section(section, workspace)
        if not text.strip():
            continue
        body.append(markdown_to_gost_typst(text, workspace / "sections", workspace / "build" / "gost" / "images", seen_images))
        body.append("")
    body.append('#bibliography("references.bib", full: true)')
    return "\n".join(body).strip() + "\n"


def markdown_to_gost_typst(markdown: str, base_dir: Path, images_dir: Path, seen_images: set[str]) -> str:
    lines: list[str] = []
    in_math = False
    math_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if _skip(line):
            continue
        if line.strip() == "$$":
            if in_math:
                formula = latex_to_typst_math("\n".join(math_lines))
                lines.append(f"$ {formula} $")
                math_lines = []
            in_math = not in_math
            continue
        if in_math:
            math_lines.append(line)
            continue
        if not line.strip():
            lines.append("")
            continue
        image = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line)
        if image:
            caption, rel = image.groups()
            out = _copy_image(base_dir, images_dir, rel, seen_images)
            if out:
                lines.append(f'#figure(image("images/{out.name}", width: 85%), caption: [{_txt(caption)}])')
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            title_raw = _strip_inline(heading.group(2))
            title = _txt(_normalize_heading_text(title_raw))
            if _is_unnumbered_block_title(title_raw):
                lines.append(f"#strong[{title}]")
            else:
                level = _heading_level(len(heading.group(1)), title_raw)
                lines.append("=" * level + " " + title)
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            lines.append("- " + _txt(_strip_inline(bullet.group(1))))
            continue
        if _unfinished(line):
            continue
        lines.append(_txt(_strip_inline(line)))
    if math_lines:
        lines.append(f"$ {latex_to_typst_math(chr(10).join(math_lines))} $")
    return "\n".join(lines).strip()


def latex_to_typst_math(text: str) -> str:
    text = _strip_inline(text)
    text = text.replace(r"\frac{|C|}{|F|}", "|C| / |F|")
    text = text.replace(r"\frac{coverage}{distance}", "coverage / distance")
    text = text.replace(r"\frac{\text{coverage}}{\text{distance}}", "coverage / distance")
    text = text.replace(r"\frac{text{coverage}}{text{distance}}", "coverage / distance")
    text = text.replace(r"\frac{\sigma(d_1, d_2, ..., d_n)}{\mu(d_1, d_2, ..., d_n)}", "sigma(d_1, d_2, ..., d_n) / mu(d_1, d_2, ..., d_n)")
    text = re.sub(r"\\?text\{([^}]+)\}", r'"\1"', text)
    text = re.sub(r"\\?frac\{([^{}]+)\}\{([^{}]+)\}", lambda m: f"{_unquote(m.group(1))} / {_unquote(m.group(2))}", text)
    text = re.sub(r"\\mathcal\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbb\{R\}", "RR", text)
    text = re.sub(r"\\min_\{([^}]+)\}", r"min_(\1)", text)
    text = re.sub(r"min_\{([^}]+)\}", r"min_(\1)", text)
    text = re.sub(r"d_\{min\}", "d_min", text)
    text = text.replace(r"\subset", "subset")
    text = text.replace(r"\sigma", "sigma")
    text = text.replace(r"\mu", "mu")
    text = text.replace(r"\|", "|")
    text = text.replace("\\", "")
    text = text.replace("d_{min}", "d_min")
    text = text.strip()
    text = re.sub(r'(?<!")\bcoverage\b(?!")', '"coverage"', text)
    text = re.sub(r'(?<!")\befficiency\b(?!")', '"efficiency"', text)
    text = re.sub(r'(?<!")\bdistance\b(?!")', '"distance"', text)
    text = re.sub(r"\bsigma\b", "sigma", text)
    text = re.sub(r"\bmu\b", "mu", text)
    text = text.replace("CV =", '"CV" =')
    text = re.sub(r"\bx_([A-Za-z])\b", r"x_\1", text)
    text = re.sub(r"\bd_([A-Za-z]+)\b", r"d_\1", text)
    return text


def _unquote(value: str) -> str:
    return value.strip().strip('"')


def build_bibtex(workspace: Path) -> str:
    records = list(SourceStore(workspace).load().values())
    defaults = [
        SourceRecord("project-readme", "Multi-Robot Coverage Research", "README.md", "project", year="2026"),
        SourceRecord("project-results", "Результаты экспериментов multi-robot coverage", "results/lab/presentation_report/report_cards.md", "project", year="2026"),
        SourceRecord("project-methods", "Методические материалы по группам методов", "docs/METHOD_GROUPS.md", "project", year="2026"),
    ]
    by_id = {r.id: r for r in defaults}
    for record in records:
        if record.source_type != "project":
            by_id[record.id] = record
    chunks = []
    for record in by_id.values():
        key = _bib_key(record)
        if record.source_type == "paper":
            chunks.append(f"""@article{{{key},
  title = {{{_bib(record.title)}}},
  author = {{{_bib(' and '.join(record.authors) or 'Unknown')}}},
  year = {{{_bib(record.year or '2026')}}},
  url = {{{_bib(record.url)}}},
}}""")
        else:
            chunks.append(f"""@misc{{{key},
  title = {{{_bib(record.title)}}},
  year = {{{_bib(record.year or '2026')}}},
  url = {{{_bib(record.url)}}},
}}""")
    return "\n\n".join(chunks) + "\n"


def validate_gost_export(main_typ: Path) -> list[str]:
    errors = []
    text = main_typ.read_text(encoding="utf-8") if main_typ.exists() else ""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            errors.append(f"Forbidden pattern: {pattern}")
    if "#show: gost.with" not in text:
        errors.append("Нет #show: gost.with(...)")
    if "#outline()" not in text:
        errors.append("Нет #outline()")
    if "#bibliography(" not in text:
        errors.append("Нет #bibliography(...)")
    if re.search(r"^={2,}\s+Глава\s+\d", text, flags=re.MULTILINE):
        errors.append("Главы не должны быть подразделами: используйте один `=` для `Глава N`.")
    if re.search(r"^=+\s+(Формулы и метрики|Иллюстрации и результаты)", text, flags=re.MULTILINE):
        errors.append("Служебные блоки формул/иллюстраций не должны попадать в нумеруемые заголовки.")
    return errors


def _copy_image(base_dir: Path, images_dir: Path, rel: str, seen: set[str]) -> Path | None:
    src = (base_dir / rel).resolve()
    if not src.exists():
        return None
    if str(src) in seen:
        return None
    seen.add(str(src))
    out = images_dir / src.name
    shutil.copy2(src, out)
    return out


def _skip(line: str) -> bool:
    low = line.lower()
    if any(item.lower() in low for item in FORBIDDEN_PATTERNS):
        return True
    return line.strip().startswith(("- `docs/", "- `coverage_", "- `results/"))


def _unfinished(line: str) -> bool:
    text = _strip_inline(line).strip()
    return len(text) > 80 and not text.endswith((".", "!", "?", "…", ":", ";", ")", "»"))


def _heading_level(markdown_level: int, title: str) -> int:
    clean = _strip_inline(title).strip()
    if re.match(r"^Глава\s+\d+", clean, flags=re.IGNORECASE):
        return 1
    if re.match(r"^\d+\.\d+", clean):
        return 2
    if clean.lower() in {"введение", "заключение", "список использованных источников"}:
        return 1
    return min(markdown_level, 3)


def _is_unnumbered_block_title(title: str) -> bool:
    return _strip_inline(title).strip().lower() in {"формулы и метрики", "иллюстрации и результаты"}


def _normalize_heading_text(title: str) -> str:
    clean = _strip_inline(title).strip()
    if re.match(r"^Глава\s+\d+", clean, flags=re.IGNORECASE):
        return clean
    # Remove duplicated manual numbering from lower-level headings; modern-g7-32 adds numbering itself.
    clean = re.sub(r"^\d+\.\d+\.\s*", "", clean)
    return clean


def _strip_inline(text: str) -> str:
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = text.replace("$", "")
    return text.strip()


def _t(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _txt(text: str) -> str:
    return _t(latex_to_typst_math(text))


def _bib(text: str) -> str:
    return str(text).replace("{", "").replace("}", "")


def _bib_key(record: SourceRecord) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "-", record.id or record.title).strip("-").lower()
    return key or "source"


def _section_sort_key(section_id: str) -> tuple[int, ...]:
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
