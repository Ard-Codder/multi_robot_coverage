from __future__ import annotations

"""
Batch runner for CoverageLab.

Reads YAML config with:
  output_dir
  scene_path
  matrix.algorithms
  matrix.seeds

Writes per-run JSON + summary.csv and renders PNG/GIF next to JSON.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_lab.io import load_scene_yaml, save_result_json
from coverage_lab.render import render_gif, save_plots
from coverage_lab.sim import run_experiment_lab
from coverage_lab.algorithms.baselines import (
    BaselineFrontier,
    BaselineGrid,
    BaselineRandomWalk,
    BaselineVoronoi,
)
from coverage_lab.algorithms.classic import Boustrophedon, DARP, STC
from coverage_lab.algorithms.mapf_wrapped import CBSDeconflictWrapper
from coverage_lab.ml_planner.guided_algo import MLGuidedPlanner
from coverage_lab.rl.ppo_policy import PpoPolicyPlanner


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _algo_factory(name: str, seed: int):
    n = name.lower().strip()
    # MAPF-wrapped variants: prefix "cbs_" (example: cbs_boustrophedon)
    if n.startswith("cbs_"):
        inner_name = n[len("cbs_") :]
        inner = _algo_factory(inner_name, seed)
        # Smaller horizon keeps CBS fast/stable on Windows
        return CBSDeconflictWrapper(inner=inner, horizon=24)
    if n == "baseline_random_walk":
        return BaselineRandomWalk(seed=seed)
    if n == "baseline_grid":
        return BaselineGrid()
    if n == "baseline_voronoi":
        return BaselineVoronoi()
    if n == "baseline_frontier":
        return BaselineFrontier(seed=seed)
    if n == "boustrophedon":
        return Boustrophedon()
    if n == "stc":
        return STC()
    if n == "darp_boustro":
        return DARP(inner="boustro")
    if n == "darp_stc":
        return DARP(inner="stc")
    if n == "ml_guided":
        # default model path (can be overridden by env var or future config expansion)
        model_path = str((ROOT / "results" / "lab" / "ml_guided" / "model.pt").resolve())
        try:
            return MLGuidedPlanner(model_path=model_path, window=9)
        except (ImportError, OSError) as e:
            print(f"[skip] ml_guided: torch/model load failed: {e}")
            return None
    if n == "ppo_policy":
        env_path = Path(str(os.environ.get("PPO_MODEL_PATH", ""))).expanduser() if os.environ.get("PPO_MODEL_PATH") else None
        candidates = []
        if env_path:
            candidates.append(env_path if env_path.is_absolute() else (ROOT / env_path).resolve())
        candidates.extend(
            [
                (ROOT / "results" / "lab" / "rl" / "ppo_dynamic_B_long" / "best" / "best_model.zip").resolve(),
                (ROOT / "results" / "lab" / "rl" / "ppo_dynamic_B_long" / "ppo_model.zip").resolve(),
            ]
        )
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = c
                break
        if model_path is None:
            print("[skip] ppo_policy: no PPO model found (set PPO_MODEL_PATH or train first)")
            return None
        try:
            return PpoPolicyPlanner(model_path=str(model_path))
        except (ImportError, OSError, FileNotFoundError) as e:
            print(f"[skip] ppo_policy: failed to load model: {e}")
            return None
    raise ValueError(f"Unknown algorithm: {name}")


def _flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "algorithm": row.get("algorithm"),
        "seed": row.get("seed"),
        "scene": row.get("scene"),
        "coverage_percent": row.get("coverage_percent"),
        "time_to_coverage_sec": row.get("time_to_coverage_sec"),
        "distance_travelled_m": row.get("distance_travelled_m"),
        "efficiency": row.get("efficiency"),
        "steps": row.get("steps"),
        "load_balance_cv": row.get("load_balance_cv"),
        "blocked_moves": row.get("blocked_moves"),
        "min_pedestrian_clearance_m": row.get("min_pedestrian_clearance_m"),
        "robot_robot_collisions": row.get("robot_robot_collisions"),
        "robot_ped_violations": row.get("robot_ped_violations"),
        "json_path": row.get("_json_path"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--render", action="store_true", help="Render PNG/GIF for each run")
    ap.add_argument("--gif-fps", type=int, default=16)
    ap.add_argument("--gif-stride", type=int, default=3)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    out_rel = Path(str(cfg.get("output_dir", "results/lab/batch")))
    output_dir = (ROOT / out_rel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_path = cfg.get("scene_path")
    if not scene_path:
        raise SystemExit("config must include scene_path")
    scene = load_scene_yaml((ROOT / Path(str(scene_path))).resolve())

    matrix = cfg.get("matrix") or {}
    algorithms: List[str] = list(matrix.get("algorithms") or [])
    seeds: List[int] = list(matrix.get("seeds") or [0])
    if not algorithms:
        raise SystemExit("matrix.algorithms empty")

    summary_rows: List[Dict[str, Any]] = []
    for algo_name in algorithms:
        for seed in seeds:
            try:
                algo = _algo_factory(algo_name, int(seed))
            except (ImportError, OSError) as e:
                # e.g. torch DLL issues on Windows Store Python; don't block the whole batch
                print(f"[skip] {algo_name} seed={seed}: {e}")
                continue
            if algo is None:
                print(f"[skip] {algo_name} seed={seed}: factory returned None")
                continue
            stem = f"{scene.name}__{algo_name}__seed{seed}"
            json_path = output_dir / f"{stem}.json"
            result = run_experiment_lab(
                algorithm=algo,
                algorithm_name=algo_name,
                scene=scene,
                seed=int(seed),
            )
            result["_json_path"] = str(json_path.relative_to(ROOT))
            save_result_json(json_path, result)
            summary_rows.append(_flatten(result))

            if args.render:
                prefix = output_dir / stem
                save_plots(result, prefix)
                render_gif(result, output_dir / f"{stem}.gif", fps=args.gif_fps, stride=args.gif_stride)

    if summary_rows:
        summary_csv = output_dir / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"OK: {len(summary_rows)} runs -> {output_dir}")


if __name__ == "__main__":
    main()

