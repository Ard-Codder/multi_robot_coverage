"""Журнал запусков агента (JSONL) для панели и воспроизводимости."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def append_agent_session_log(root: Path, task: str, result: Dict[str, Any]) -> None:
    log_dir = root / "results" / "agent_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "sessions.jsonl"
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "task_excerpt": task[:2000],
        "iteration": result.get("iteration"),
        "done": result.get("done"),
        "max_iterations": result.get("max_iterations"),
    }
    if "_error" in result:
        entry["error"] = result["_error"]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
