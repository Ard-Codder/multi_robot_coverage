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
import json
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
from coverage_lab.ml_planner.goal_guided_algo import MLGoalPlanner
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
    if n in {"ml_guided", "ml_guided_pure", "ml_guided_soft_guarded", "ml_guided_guarded"}:
        model_env = os.environ.get("ML_MODEL_PATH")
        model_path = str((ROOT / model_env).resolve()) if model_env else str((ROOT / "results" / "lab" / "ml_guided" / "model.pt").resolve())
        if not Path(model_path).exists():
            print(f"[skip] {name}: model not found at {model_path}; train with experiments_lab/train_ml_guided.py or set ML_MODEL_PATH")
            return None
        mode = {
            "ml_guided": "soft_guarded",
            "ml_guided_pure": "pure",
            "ml_guided_soft_guarded": "soft_guarded",
            "ml_guided_guarded": "guarded",
        }[n]
        try:
            return MLGuidedPlanner(model_path=model_path, window=9, mode=mode)
        except (ImportError, OSError) as e:
            print(f"[skip] {name}: torch/model load failed: {e}")
            return None
    if n in {
        "ml_goal",
        "ml_goal_pure",
        "ml_goal_soft_guarded",
        "ml_goal_guarded",
        "ml_goal_ranked",
        "ml_goal_allocated",
    }:
        model_env = os.environ.get("ML_GOAL_MODEL_PATH")
        model_path = str((ROOT / model_env).resolve()) if model_env else str((ROOT / "results" / "lab" / "ml_goal" / "model.pt").resolve())
        if not Path(model_path).exists():
            print(f"[skip] {name}: goal model not found at {model_path}; train with experiments_lab/train_ml_goal.py or set ML_GOAL_MODEL_PATH")
            return None
        mode = {
            "ml_goal": "soft_guarded",
            "ml_goal_pure": "pure",
            "ml_goal_soft_guarded": "soft_guarded",
            "ml_goal_guarded": "guarded",
            "ml_goal_ranked": "soft_guarded",
            "ml_goal_allocated": "soft_guarded",
        }[n]
        try:
            return MLGoalPlanner(
                model_path=model_path,
                mode=mode,
                top_k=24 if n in {"ml_goal_ranked", "ml_goal_allocated"} else 16,
                use_allocation=n != "ml_goal_ranked",
            )
        except (ImportError, OSError) as e:
            print(f"[skip] {name}: torch/model load failed: {e}")
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
        "ml_mode": row.get("ml_mode"),
        "ml_input_mode": row.get("ml_input_mode"),
        "ml_model_variant": row.get("ml_model_variant"),
        "ml_total_decisions": row.get("ml_total_decisions"),
        "ml_model_steps": row.get("ml_model_steps"),
        "ml_fallback_steps": row.get("ml_fallback_steps"),
        "ml_model_step_rate": row.get("ml_model_step_rate"),
        "ml_fallback_rate": row.get("ml_fallback_rate"),
        "ml_unsafe_model_targets": row.get("ml_unsafe_model_targets"),
        "ml_action_counts": json.dumps(row.get("ml_action_counts") or {}, ensure_ascii=False),
        "ml_goal_mode": row.get("ml_goal_mode"),
        "ml_goal_total_decisions": row.get("ml_goal_total_decisions"),
        "ml_goal_model_steps": row.get("ml_goal_model_steps"),
        "ml_goal_fallback_steps": row.get("ml_goal_fallback_steps"),
        "ml_goal_model_rate": row.get("ml_goal_model_rate"),
        "ml_goal_fallback_rate": row.get("ml_goal_fallback_rate"),
        "ml_goal_unsafe_goals": row.get("ml_goal_unsafe_goals"),
        "ml_goal_avg_l1_distance": row.get("ml_goal_avg_l1_distance"),
        "ml_goal_allocated_steps": row.get("ml_goal_allocated_steps"),
        "ml_goal_reserved_conflicts": row.get("ml_goal_reserved_conflicts"),
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
    seeds: List[int] = list(matrix.get("seeds") or [])
    sr = matrix.get("seeds_range")
    if isinstance(sr, (list, tuple)) and len(sr) == 2:
        seeds = list(range(int(sr[0]), int(sr[1])))
    if not seeds:
        seeds = [0]
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
            print(
                f"[run] {algo_name} seed={seed} steps={result.get('steps')} "
                f"cov={float(result.get('coverage_percent') or 0):.2f}",
                flush=True,
            )

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

