"""
Пакетный запуск экспериментов по матрице (алгоритм × seed) и сводная таблица CSV.

Пример:
  python experiments/run_batch.py
  python experiments/run_batch.py --config experiments/batch_default.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_sim.simulation.simulation_loop import run_experiment


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _git_head(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo),
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _flatten_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """Поля для CSV: скаляры и пути."""
    out = {
        "algorithm": row.get("algorithm"),
        "seed": row.get("experiment_seed"),
        "coverage_percent": row.get("coverage_percent"),
        "time_to_coverage_sec": row.get("time_to_coverage_sec"),
        "distance_travelled_m": row.get("distance_travelled_m"),
        "efficiency": row.get("efficiency"),
        "steps": row.get("steps"),
        "load_balance_cv": row.get("load_balance_cv"),
        "distance_mean_m": row.get("distance_mean_m"),
        "distance_std_m": row.get("distance_std_m"),
        "blocked_moves": row.get("blocked_moves"),
        "use_isaac_sim": row.get("use_isaac_sim"),
        "target_coverage": row.get("target_coverage"),
        "max_steps": row.get("max_steps"),
        "json_path": row.get("_json_path"),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch coverage experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "batch_default.yaml",
        help="YAML с матрицей algorithms × seeds",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    out_rel = Path(str(cfg.get("output_dir", "results/batch")))
    output_dir = (ROOT / out_rel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = cfg.get("matrix") or {}
    algorithms: List[str] = list(matrix.get("algorithms") or [])
    seeds: List[int] = list(matrix.get("seeds") or [0])
    settings = cfg.get("settings") or {}
    max_steps = settings.get("max_steps")
    target_coverage = settings.get("target_coverage")
    save_plots = bool(settings.get("save_plots", False))
    use_isaac = settings.get("use_isaac_sim")
    use_isaac_bool = bool(use_isaac) if use_isaac is not None else None

    if not algorithms:
        raise SystemExit("matrix.algorithms пуст — укажите список алгоритмов в YAML")

    summary_rows: List[Dict[str, Any]] = []

    for algo in algorithms:
        for seed in seeds:
            stem = f"{algo}_seed{seed}"
            json_path = output_dir / f"{stem}.json"
            plot_prefix = output_dir / stem
            result = run_experiment(
                algorithm_name=algo,
                output_json=json_path,
                output_plot_prefix=plot_prefix,
                root_dir=ROOT,
                keep_visualization_open_sec=0.0,
                require_isaac=False,
                experiment_seed=int(seed),
                max_steps=int(max_steps) if max_steps is not None else None,
                target_coverage=float(target_coverage) if target_coverage is not None else None,
                save_plots=save_plots,
                use_isaac_sim=use_isaac_bool,
            )
            result["_json_path"] = str(json_path.relative_to(ROOT))
            summary_rows.append(_flatten_summary(result))

    summary_csv = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)

    meta = {
        "config": str(args.config),
        "output_dir": str(out_rel),
        "runs": len(summary_rows),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "git_head": _git_head(ROOT),
        },
    }
    (output_dir / "batch_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Готово: {len(summary_rows)} прогонов, сводка: {summary_csv}")


if __name__ == "__main__":
    main()
