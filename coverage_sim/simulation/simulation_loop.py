from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from coverage_sim.algorithms import (
    FrontierCoverage,
    GridCoverage,
    RandomWalkCoverage,
    VoronoiCoverage,
    ZonalRandomWalkCoverage,
)
from coverage_sim.env.isaac_env import IsaacEnvironment
from coverage_sim.metrics.coverage_metrics import CoverageMetrics
from coverage_sim.robots.robot_state import RobotState
from coverage_sim.simulation.spawn_robots import create_spawn_points
from coverage_sim.visualization.coverage_map import save_plots


def run_experiment(
    algorithm_name: str,
    output_json: Path,
    output_plot_prefix: Path,
    root_dir: Path,
    keep_visualization_open_sec: float = 0.0,
    require_isaac: bool = False,
    *,
    experiment_seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    target_coverage: Optional[float] = None,
    robot_count: Optional[int] = None,
    save_plots: bool = True,
    use_isaac_sim: Optional[bool] = None,
) -> Dict[str, Any]:
    robots_cfg = _load_yaml(root_dir / "coverage_sim" / "configs" / "robots.yaml")
    world_cfg = _load_yaml(root_dir / "coverage_sim" / "configs" / "world.yaml")

    world = world_cfg["world"]
    if use_isaac_sim is not None:
        world = {**world, "use_isaac_sim": bool(use_isaac_sim)}
    sim_cfg = world_cfg["simulation"]
    cov_cfg = world_cfg["coverage"]

    bounds = (
        float(world["bounds_xy"]["x_min"]),
        float(world["bounds_xy"]["x_max"]),
        float(world["bounds_xy"]["y_min"]),
        float(world["bounds_xy"]["y_max"]),
    )
    dt_sec = float(sim_cfg["dt_sec"])
    max_steps_eff = int(max_steps) if max_steps is not None else int(sim_cfg["max_steps"])
    num_robots = int(robot_count) if robot_count is not None else int(robots_cfg["robots"]["count"])
    target_cov = float(target_coverage) if target_coverage is not None else float(cov_cfg["target_coverage"])
    rng_seed = experiment_seed if experiment_seed is not None else None

    env = IsaacEnvironment(
        bounds_xy=bounds,
        dt_sec=dt_sec,
        use_isaac_sim=bool(world["use_isaac_sim"]),
        require_isaac=require_isaac,
        rng_seed=rng_seed,
    )
    env.load_world(str(world["scene_name"]))
    # Если Isaac Sim не используется (или недоступен), задаём препятствия из YAML.
    # В режиме с Isaac препятствия извлекаются из USD сцены.
    if not env.use_isaac_sim:
        fb = world_cfg.get("fallback_obstacles") or []
        disks = [
            (float(o["x"]), float(o["y"]), float(o["r"]))
            for o in fb
            if isinstance(o, dict) and "x" in o and "y" in o and "r" in o
        ]
        if disks:
            env.set_static_obstacles(disks)
    env.configure_pedestrians(world_cfg.get("pedestrians"))

    robot_states = _spawn_robots(env, num_robots, bounds)
    algorithm = _build_algorithm(
        algorithm_name,
        bounds,
        float(cov_cfg["grid_resolution_m"]),
        experiment_seed=experiment_seed,
    )
    grid_res = float(cov_cfg["grid_resolution_m"])
    metrics = CoverageMetrics(bounds_xy=bounds, grid_resolution_m=grid_res)
    metrics.set_blocked_cells(set(env.get_obstacle_cells(grid_resolution_m=grid_res, bounds_xy=bounds)))
    robot_paths: Dict[str, list[list[float]]] = {name: [] for name in robot_states}
    pedestrian_paths: Dict[str, list[list[float]]] = {}
    for p in env._pedestrians:
        pedestrian_paths[p.pid] = []
    min_ped_clear: Optional[float] = None

    steps_run = 0
    for _ in range(max_steps_eff):
        steps_run += 1
        targets = algorithm.choose_targets(robot_states)
        for name, target in targets.items():
            robot_states[name].target = target.copy()
            env.set_robot_target(name, target)
        env.step_simulation()

        for name, state in robot_states.items():
            state.update_position(env.get_robot_pose(name))
            robot_paths[name].append([float(state.position[0]), float(state.position[1])])
        for p in env._pedestrians:
            pedestrian_paths[p.pid].append(
                [float(p.position[0]), float(p.position[1])]
            )
        if env._pedestrians:
            mc = env.min_robot_pedestrian_clearance_m()
            min_ped_clear = mc if min_ped_clear is None else min(min_ped_clear, mc)

        metrics.update(robot_states)

        if metrics.coverage_percent() >= target_cov:
            break

    result = metrics.to_dict(
        robot_states,
        dt_sec=dt_sec,
        target_coverage_for_ttc=target_cov,
    )
    result["algorithm"] = algorithm_name
    result["steps"] = steps_run
    result["experiment_seed"] = experiment_seed
    result["target_coverage"] = target_cov
    result["max_steps"] = max_steps_eff
    result["use_isaac_sim"] = bool(world["use_isaac_sim"])
    result["robot_paths"] = robot_paths
    result["obstacles"] = [
        {"x": x, "y": y, "r": r} for x, y, r in env.get_obstacle_disks()
    ]
    result["obstacle_cells"] = [[c[0], c[1]] for c in env.get_obstacle_cells()]
    result["blocked_moves"] = env.get_blocked_moves()
    result["pedestrians_enabled"] = bool(env._pedestrians)
    result["pedestrian_paths"] = pedestrian_paths
    result["min_pedestrian_clearance_m"] = (
        float(min_ped_clear) if min_ped_clear is not None and env._pedestrians else None
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_plots:
        save_plots(result, output_plot_prefix)
    if keep_visualization_open_sec > 0.0:
        env.keep_alive_for_seconds(keep_visualization_open_sec)
    env.close()
    return result


def _spawn_robots(
    env: IsaacEnvironment,
    num_robots: int,
    bounds: Tuple[float, float, float, float],
) -> Dict[str, RobotState]:
    points = env.suggest_spawn_points(num_robots=num_robots, bounds_xy=bounds)
    if not points:
        points = create_spawn_points(num_robots=num_robots, bounds_xy=bounds)
    states: Dict[str, RobotState] = {}
    for idx in range(num_robots):
        name = f"robot_{idx}"
        x, y = float(points[idx][0]), float(points[idx][1])
        x, y = env.project_to_free(x, y)
        env.spawn_robot(name=name, x=x, y=y)
        pos = env.get_robot_pose(name)
        states[name] = RobotState(name=name, position=pos.copy(), target=pos.copy(), last_position=pos.copy())
    return states


def _build_algorithm(
    name: str,
    bounds: Tuple[float, float, float, float],
    grid_resolution: float,
    *,
    experiment_seed: Optional[int] = None,
):
    algo = name.lower().strip()
    seed = experiment_seed
    if algo == "random_walk":
        return RandomWalkCoverage(bounds_xy=bounds, seed=seed if seed is not None else 42)
    if algo == "grid":
        return GridCoverage(bounds_xy=bounds)
    if algo == "voronoi":
        return VoronoiCoverage(bounds_xy=bounds)
    if algo == "frontier":
        return FrontierCoverage(
            bounds_xy=bounds,
            grid_resolution_m=grid_resolution,
            seed=seed if seed is not None else 7,
        )
    if algo == "zonal":
        return ZonalRandomWalkCoverage(bounds_xy=bounds, seed=seed if seed is not None else 12)
    raise ValueError(f"Unknown algorithm: {name}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

