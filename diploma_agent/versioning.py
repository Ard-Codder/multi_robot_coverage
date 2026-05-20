"""Private timestamped snapshots and diffs for thesis workspaces."""

from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from diploma_agent import config
from diploma_agent.page_budget import CHARS_PER_PAGE_TARGET
from diploma_agent.state import ensure_workspace, now_iso


@dataclass(frozen=True)
class VersionInfo:
    id: str
    path: Path
    label: str
    created_at: str
    target_pages: int
    chars: int
    words: int
    approx_pages: float


def create_snapshot(
    workspace: Path | None = None,
    label: str = "",
    reason: str = "",
    target_pages: int = 80,
    run_log: list[dict] | None = None,
) -> VersionInfo:
    root = ensure_workspace(workspace or config.workspace_dir())
    stamp = now_iso().replace(":", "-").replace("+", "_")
    safe_label = _safe_name(label or "snapshot")
    version_dir = root / "versions" / f"{stamp}_{safe_label}"
    version_dir.mkdir(parents=True, exist_ok=False)

    for dirname in ("sections", "build", "quality"):
        src = root / dirname
        if src.exists():
            shutil.copytree(src, version_dir / dirname, dirs_exist_ok=True)

    stats = workspace_stats(root)
    manifest = {
        "id": version_dir.name,
        "created_at": now_iso(),
        "label": label or version_dir.name,
        "reason": reason,
        "target_pages": target_pages,
        "build_files": _build_file_sizes(version_dir),
        **stats,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if run_log is not None:
        (version_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return _info_from_manifest(version_dir)


def list_versions(workspace: Path | None = None) -> list[VersionInfo]:
    root = ensure_workspace(workspace or config.workspace_dir())
    versions = []
    for path in sorted((root / "versions").glob("*")):
        if path.is_dir() and (path / "manifest.json").exists():
            versions.append(_info_from_manifest(path))
    return versions


def compare_versions(
    old_id: str = "previous",
    new_id: str = "latest",
    workspace: Path | None = None,
) -> Path:
    root = ensure_workspace(workspace or config.workspace_dir())
    old_path = _resolve_version(old_id, root)
    new_path = _resolve_version(new_id, root)
    lines = [f"# Diff {old_path.name} -> {new_path.name}", ""]
    old_sections = {p.name: p for p in (old_path / "sections").glob("*.md")}
    new_sections = {p.name: p for p in (new_path / "sections").glob("*.md")}
    for name in sorted(set(old_sections) | set(new_sections)):
        old_text = old_sections.get(name).read_text(encoding="utf-8").splitlines() if name in old_sections else []
        new_text = new_sections.get(name).read_text(encoding="utf-8").splitlines() if name in new_sections else []
        added = max(len(new_text) - len(old_text), 0)
        removed = max(len(old_text) - len(new_text), 0)
        changed = old_text != new_text
        lines.append(f"## {name}")
        lines.append(f"- changed: {changed}")
        lines.append(f"- old lines: {len(old_text)}")
        lines.append(f"- new lines: {len(new_text)}")
        lines.append(f"- added lines approx: {added}")
        lines.append(f"- removed lines approx: {removed}")
        if changed:
            diff = difflib.unified_diff(old_text, new_text, fromfile=f"old/{name}", tofile=f"new/{name}", lineterm="")
            lines.append("")
            lines.append("```diff")
            lines.extend(list(diff)[:240])
            lines.append("```")
        lines.append("")
    diff_path = root / "versions" / f"diff_{old_path.name}_to_{new_path.name}.md"
    diff_path.write_text("\n".join(lines), encoding="utf-8")
    return diff_path


def restore_version(version_id: str, workspace: Path | None = None) -> Path:
    root = ensure_workspace(workspace or config.workspace_dir())
    version = _resolve_version(version_id, root)
    for dirname in ("sections", "build", "quality"):
        src = version / dirname
        dst = root / dirname
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return version


def prune_versions(
    workspace: Path | None = None,
    keep_labels: list[str] | None = None,
    keep_latest: bool = True,
    dry_run: bool = False,
) -> list[str]:
    root = ensure_workspace(workspace or config.workspace_dir())
    versions = list_versions(root)
    keep_labels = keep_labels or []
    keep_ids = set()
    for item in versions:
        if any(label and (label in item.id or label in item.label) for label in keep_labels):
            keep_ids.add(item.id)
    if keep_latest and versions:
        keep_ids.add(versions[-1].id)

    removed: list[str] = []
    for item in versions:
        if item.id in keep_ids:
            continue
        removed.append(item.id)
        if not dry_run:
            shutil.rmtree(item.path)

    if not dry_run:
        for diff in (root / "versions").glob("diff_*.md"):
            diff.unlink(missing_ok=True)
    return removed


def workspace_stats(workspace: Path | None = None) -> dict:
    root = ensure_workspace(workspace or config.workspace_dir())
    chars = 0
    words = 0
    files = 0
    for path in (root / "sections").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        chars += len(text)
        words += len(text.split())
        files += 1
    return {
        "chars": chars,
        "words": words,
        "section_files": files,
        "approx_pages": round(chars / CHARS_PER_PAGE_TARGET, 1),
    }


def _build_file_sizes(version_dir: Path) -> dict[str, int]:
    build = version_dir / "build"
    if not build.exists():
        return {}
    return {path.name: path.stat().st_size for path in build.iterdir() if path.is_file()}


def _info_from_manifest(path: Path) -> VersionInfo:
    raw = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return VersionInfo(
        id=raw.get("id", path.name),
        path=path,
        label=raw.get("label", path.name),
        created_at=raw.get("created_at", ""),
        target_pages=int(raw.get("target_pages", 0)),
        chars=int(raw.get("chars", 0)),
        words=int(raw.get("words", 0)),
        approx_pages=float(raw.get("approx_pages", 0.0)),
    )


def _resolve_version(version_id: str, root: Path) -> Path:
    versions = list_versions(root)
    if not versions:
        raise ValueError("Нет сохраненных версий.")
    if version_id == "latest":
        return versions[-1].path
    if version_id == "previous":
        if len(versions) < 2:
            raise ValueError("Нужны хотя бы две версии для previous.")
        return versions[-2].path
    for item in versions:
        if item.id == version_id or item.label == version_id:
            return item.path
    raise ValueError(f"Версия не найдена: {version_id}")


def _safe_name(value: str) -> str:
    allowed = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return allowed.strip("_")[:80] or "snapshot"
