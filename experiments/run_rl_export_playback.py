"""
Прогон упрощённой RL-среды (GridCoverageGymEnv) и экспорт JSON в формате, совместимом с viz/playback.html.

Пример:
  python experiments/run_rl_export_playback.py
  python experiments/run_rl_export_playback.py --policy random --seed 2 --output results/rl_playback.json

Политика:
  random — случайное действие (демо для просмотрщика)
  center — всегда смещение цели к центру (1,1) в MultiDiscrete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_sim.rl.grid_coverage_env import GridCoverageGymEnv  # noqa: E402


def _append_snapshot(paths: dict[str, list[list[float]]], env: GridCoverageGymEnv) -> None:
    for i, (wx, wy) in enumerate(env.get_robot_positions_world()):
        paths[f"robot_{i}"].append([float(wx), float(wy)])


def main() -> None:
    p = argparse.ArgumentParser(description="RL grid rollout → JSON для viz/playback.html")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "rl_playback_export.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--target-coverage", type=float, default=0.88)
    p.add_argument("--n-robots", type=int, default=3)
    p.add_argument("--policy", choices=("random", "center"), default="random")
    args = p.parse_args()

    env = GridCoverageGymEnv(
        grid_size=args.grid_size,
        target_coverage=args.target_coverage,
        max_steps=args.max_steps,
        n_robots=args.n_robots,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)

    paths: dict[str, list[list[float]]] = {f"robot_{i}": [] for i in range(env.num_robots)}
    coverage_history: list[float] = []

    obs, _ = env.reset(seed=args.seed)
    coverage_history.append(float(obs["coverage"][0]))
    _append_snapshot(paths, env)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        if args.policy == "center":
            action = np.array([1, 1], dtype=np.int64)
        else:
            action = np.array(
                [int(rng.integers(0, int(n))) for n in env.action_space.nvec],
                dtype=np.int64,
            )

        obs, _r, terminated, truncated, _ = env.step(action)
        steps += 1
        coverage_history.append(float(obs["coverage"][0]))
        _append_snapshot(paths, env)

    out = {
        "algorithm": f"rl_grid_{args.policy}",
        "experiment_seed": args.seed,
        "coverage_percent": float(obs["coverage"][0]),
        "coverage_history": coverage_history,
        "robot_paths": paths,
        "obstacles": [],
        "use_isaac_sim": False,
        "steps": steps,
        "target_coverage": args.target_coverage,
        "max_steps": args.max_steps,
        "note": "Упрощённая сетка (GridCoverageGymEnv), не Isaac. Для просмотра: viz/playback.html",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Записано: {args.output} ({steps} шагов, coverage={out['coverage_percent']:.3f})")


if __name__ == "__main__":
    main()
