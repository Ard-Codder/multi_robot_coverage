"""Typst export for the local thesis workspace."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.state import ThesisState, ensure_workspace, read_section


@dataclass(frozen=True)
class RenderResult:
    typ_path: Path
    pdf_path: Path
    compiled: bool
    message: str


def _escape(text: str) -> str:
    return _escape_text(text)


def _escape_text(text: str) -> str:
    text = _pretty_formula(text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("$", "")
    for char in ("\\", "#", "*", "_", "[", "]"):
        text = text.replace(char, "\\" + char)
    return text


def _md_heading_to_typst(line: str) -> str:
    match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if not match:
        return _escape_text(line)
    level = len(match.group(1))
    return "=" * level + " " + _escape_text(match.group(2))


def markdown_to_typst(markdown: str, seen_images: set[str] | None = None) -> str:
    seen_images = seen_images if seen_images is not None else set()
    lines = []
    in_code = False
    in_math = False
    math_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if _should_skip_line(line):
            continue
        if line.strip() == "$$":
            if in_math:
                formula = _escape_text(_pretty_formula("\n".join(math_lines)))
                lines.append(f'#align(center)[#text(style: "italic")[{formula}]]')
                math_lines = []
            in_math = not in_math
            continue
        if in_math:
            math_lines.append(line)
            continue
        if line.startswith("```"):
            in_code = not in_code
            lines.append("```")
            continue
        if in_code:
            lines.append(_escape_text(line))
            continue
        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line)
        if image_match:
            caption, path = image_match.groups()
            if path in seen_images:
                continue
            seen_images.add(path)
            caption = _escape_text(caption or Path(path).name)
            lines.append(f'#figure(image("{path}", width: 85%), caption: [{caption}])')
            continue
        list_match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$", line)
        if list_match:
            lines.append("- " + _escape_text(list_match.group(1)))
            continue
        if _looks_unfinished(line):
            continue
        lines.append(_md_heading_to_typst(line))
    if math_lines:
        formula = _escape_text(_pretty_formula("\n".join(math_lines)))
        lines.append(f'#align(center)[#text(style: "italic")[{formula}]]')
    return "\n".join(lines).strip()


def _should_skip_line(line: str) -> bool:
    stripped = line.strip()
    forbidden = (
        "автоматически подготовленный",
        "Текст требует ручной",
        "Ключевые опорные материалы",
        "ТРЕБУЕТ УТОЧНЕНИЯ",
        "[ТРЕБУЕТ УТОЧНЕНИЯ]",
    )
    if any(item.lower() in stripped.lower() for item in forbidden):
        return True
    if stripped.startswith("- `docs/") or stripped.startswith("- `coverage_") or stripped.startswith("- `results/"):
        return True
    return False


def _looks_unfinished(line: str) -> bool:
    stripped = _escape_text(line).strip()
    if len(stripped) < 80:
        return False
    return not stripped.endswith((".", "!", "?", "…", ":", ";", ")", "»"))


def _pretty_formula(text: str) -> str:
    replacements = {
        r"\frac{|C|}{|F|}": "|C| / |F|",
        r"\frac{coverage}{distance}": "coverage / distance",
        r"\frac{\text{coverage}}{\text{distance}}": "coverage / distance",
        r"\frac{\sigma(d_1, d_2, ..., d_n)}{\mu(d_1, d_2, ..., d_n)}": "sigma(d1, d2, ..., dn) / mu(d1, d2, ..., dn)",
        r"\min_{r,p,t}": "min_{r,p,t}",
        r"\|": "||",
        r"\sigma": "sigma",
        r"\mu": "mu",
        r"\text{efficiency}": "efficiency",
        r"\text{coverage}": "coverage",
        r"\text{distance}": "distance",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_typst_document(state: ThesisState, workspace: Path | None = None) -> str:
    root = workspace or config.workspace_dir()
    body: list[str] = [
        '#set document(title: "Дипломная работа")',
        "#set text(font: \"Times New Roman\", size: 14pt)",
        "#set page(paper: \"a4\", margin: (left: 30mm, right: 15mm, top: 20mm, bottom: 20mm), numbering: \"1\")",
        "#set par(justify: true, first-line-indent: 1.25cm, leading: 0.65em)",
        "#set heading(numbering: \"1.1\")",
        "",
        "#align(center)[",
        "#text(size: 16pt, weight: \"bold\")[ДИПЛОМНАЯ РАБОТА]",
        "",
        f"#text(size: 14pt)[{_escape(state.topic)}]",
        "",
        "#v(1fr)",
        "#text(size: 14pt)[Москва, 2026]",
        "]",
        "",
        "#pagebreak()",
        "#outline(title: [Содержание])",
        "#pagebreak()",
        "",
    ]
    content_added = False
    seen_images: set[str] = set()
    for section_id in sorted(state.sections, key=_section_sort_key):
        section = state.sections[section_id]
        text = read_section(section, workspace)
        if not text.strip():
            continue
        body.append(markdown_to_typst(text, seen_images))
        body.append("")
        content_added = True
    if not content_added:
        body.append(markdown_to_typst(state.plan_markdown, seen_images))
    bibliography = root / "build" / "bibliography.md"
    if bibliography.exists():
        body.append(markdown_to_typst(bibliography.read_text(encoding="utf-8"), seen_images))
    return "\n".join(body).strip() + "\n"


def render(state: ThesisState, workspace: Path | None = None, compile_pdf: bool = True) -> RenderResult:
    root = ensure_workspace(workspace or config.workspace_dir())
    build = root / "build"
    typ_path = build / "diploma.typ"
    pdf_path = build / "diploma.pdf"
    typ_path.write_text(build_typst_document(state, root), encoding="utf-8")
    typst = _find_typst()
    if not compile_pdf:
        return RenderResult(typ_path, pdf_path, False, "Typst файл собран, компиляция отключена.")
    if not typst:
        return RenderResult(typ_path, pdf_path, False, "Typst файл собран. Установи typst CLI, чтобы получить PDF.")
    try:
        subprocess.run(
            [typst, "compile", "--root", str(config.ROOT), str(typ_path), str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return RenderResult(typ_path, pdf_path, False, f"Typst compile failed: {msg}")
    return RenderResult(typ_path, pdf_path, True, "PDF собран.")


def _find_typst() -> str | None:
    direct = shutil.which("typst")
    if direct:
        return direct
    user_path = config.env_str("LOCALAPPDATA", "")
    if not user_path:
        return None
    winget_root = Path(user_path) / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(winget_root.glob("Typst.Typst_*/*/typst.exe"))
    return str(matches[-1]) if matches else None


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
