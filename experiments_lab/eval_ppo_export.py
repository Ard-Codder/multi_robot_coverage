from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("experiments_lab/scenes/dynamic_B.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=Path, required=True, help="Path to PPO model .zip")
    ap.add_argument("--out", type=Path, default=Path("results/lab/rl/ppo_rollout.json"))
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    from coverage_lab.io import load_scene_yaml, save_result_json
    from coverage_lab.rl.env_gym import CoveragePedEnv

    scene = load_scene_yaml((ROOT / args.scene).resolve())
    env = CoveragePedEnv(scene=scene, seed=int(args.seed))
    model = PPO.load(str((ROOT / args.model).resolve()))

    obs, _ = env.reset(seed=int(args.seed))
    robot_paths = {f"robot_{i}": [] for i in range(scene.num_robots)}
    ped_paths = {p.id: [] for p in scene.pedestrians}
    cov_hist = []
    dist_hist = []

    steps = int(args.max_steps) if args.max_steps is not None else int(scene.max_steps)
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # capture from env internals
        for i in range(scene.num_robots):
            p = env.state.robot_xy[f"robot_{i}"]
            robot_paths[f"robot_{i}"].append([float(p[0]), float(p[1])])
        for ped in env.state.pedestrians:
            ped_paths[ped.pid].append([float(ped.position[0]), float(ped.position[1])])

        cov_hist.append(float(env.metrics.coverage_percent()))
        dist_hist.append(float(env.metrics.total_distance()))

        if terminated or truncated:
            break

    # build result contract JSON
    result: Dict[str, Any] = {
        "algorithm": "rl_ppo",
        "seed": int(args.seed),
        "scene": scene.name,
        "bounds_xy": list(scene.bounds_xy),
        "dt_sec": float(scene.dt_sec),
        "steps": int(len(cov_hist)),
        "max_steps": int(scene.max_steps),
        "grid_resolution_m": float(scene.grid_resolution_m),
        "obstacles": [{"x": o.x, "y": o.y, "r": o.r} for o in scene.obstacles],
        "blocked_moves": int(env.state.blocked_moves),
        "coverage_history": cov_hist,
        "distance_history": dist_hist,
        "visited_cells": [[int(c[0]), int(c[1])] for c in sorted(env.metrics.visited_cells)],
        "coverage_percent": float(env.metrics.coverage_percent()),
        "time_to_coverage_sec": env.metrics.time_to_coverage_sec(scene.dt_sec, float(scene.target_coverage)),
        "distance_travelled_m": float(env.metrics.total_distance()),
        "efficiency": float(env.metrics.efficiency()),
        "load_balance_cv": float(env.metrics.load_balance_cv()),
        "per_robot_distance_m": {k: float(v.travelled_distance_m) for k, v in env.metrics.robot_kpi.items()},
        "pedestrians_enabled": bool(scene.pedestrians),
        "pedestrian_paths": ped_paths,
        "min_pedestrian_clearance_m": float(env.state.min_ped_clearance_m)
        if env.state.min_ped_clearance_m is not None and scene.pedestrians
        else None,
        "robot_robot_collisions": int(env.state.robot_robot_collisions),
        "robot_ped_violations": int(env.state.robot_ped_violations),
        "robot_paths": robot_paths,
    }
    out_path = (ROOT / args.out).resolve()
    save_result_json(out_path, result)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

