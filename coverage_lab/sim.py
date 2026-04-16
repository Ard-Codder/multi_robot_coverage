from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from coverage_lab.env.grid_world import GridWorld2D, WorldState
from coverage_lab.metrics import CoverageMetricsLab
from coverage_lab.types import LabScene


@dataclass
class SimResult:
    result: Dict[str, Any]


def run_experiment_lab(
    *,
    algorithm,
    algorithm_name: str,
    scene: LabScene,
    seed: int,
    keep_paths: bool = True,
) -> Dict[str, Any]:
    world = GridWorld2D(
        bounds_xy=scene.bounds_xy,
        dt_sec=scene.dt_sec,
        grid_resolution_m=scene.grid_resolution_m,
        obstacles=scene.obstacles,
        rectangles=scene.rectangles,
        pedestrians=scene.pedestrians,
        seed=seed,
    )

    spawn = world.sample_spawn_points(scene.num_robots)
    robot_xy = {f"robot_{i}": spawn[i].copy() for i in range(scene.num_robots)}
    robot_target_xy = {k: v.copy() for k, v in robot_xy.items()}

    state = WorldState(
        robot_xy=robot_xy,
        robot_target_xy=robot_target_xy,
        pedestrians=world.pedestrians,
        blocked_moves=0,
        robot_robot_collisions=0,
        robot_ped_violations=0,
        min_ped_clearance_m=None,
    )

    metrics = CoverageMetricsLab(bounds_xy=scene.bounds_xy, grid_resolution_m=scene.grid_resolution_m)
    metrics.set_blocked_cells(set(world.obstacle_cells()))

    robot_paths: Dict[str, list[list[float]]] = {k: [] for k in robot_xy}
    ped_paths: Dict[str, list[list[float]]] = {p.pid: [] for p in state.pedestrians}

    # Initial update
    metrics.update(state.robot_xy)

    steps = 0
    for _ in range(int(scene.max_steps)):
        steps += 1
        # algorithms may optionally use metrics (visited/blocked) to be coverage-driven
        try:
            targets = algorithm.choose_targets(
                state.robot_xy,
                scene=scene,
                world=world,
                metrics=metrics,
                state=state,
            )
        except TypeError:
            # Backward compatibility for algorithms that do not accept state kwarg.
            targets = algorithm.choose_targets(state.robot_xy, scene=scene, world=world, metrics=metrics)

        # measure motion distance for KPI (approx using pre/post)
        prev = {k: v.copy() for k, v in state.robot_xy.items()}
        state = world.step(state, targets)
        for name, p0 in prev.items():
            p1 = state.robot_xy[name]
            metrics.add_robot_motion(name, float(np.linalg.norm(p1 - p0)))

        metrics.update(state.robot_xy)

        if keep_paths:
            for name, p in state.robot_xy.items():
                robot_paths[name].append([float(p[0]), float(p[1])])
            for ped in state.pedestrians:
                ped_paths[ped.pid].append([float(ped.position[0]), float(ped.position[1])])

        if metrics.coverage_percent() >= float(scene.target_coverage):
            break

    ttc = metrics.time_to_coverage_sec(scene.dt_sec, float(scene.target_coverage))
    res: Dict[str, Any] = {
        "algorithm": algorithm_name,
        "seed": int(seed),
        "scene": scene.name,
        "bounds_xy": list(scene.bounds_xy),
        "dt_sec": float(scene.dt_sec),
        "steps": int(steps),
        "max_steps": int(scene.max_steps),
        "grid_resolution_m": float(scene.grid_resolution_m),
        "obstacles": (
            [{"type": "disk", "x": o.x, "y": o.y, "r": o.r} for o in scene.obstacles]
            + [{"type": "rect", "x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in scene.rectangles]
        ),
        "obstacle_cells": [[int(c[0]), int(c[1])] for c in world.obstacle_cells()],
        "blocked_moves": int(state.blocked_moves),
        "coverage_history": metrics.coverage_history,
        "distance_history": metrics.distance_history,
        "visited_cells": [[int(c[0]), int(c[1])] for c in sorted(metrics.visited_cells)],
        "coverage_percent": float(metrics.coverage_percent()),
        "time_to_coverage_sec": float(ttc) if ttc is not None else None,
        "distance_travelled_m": float(metrics.total_distance()),
        "efficiency": float(metrics.efficiency()),
        "load_balance_cv": float(metrics.load_balance_cv()),
        "per_robot_distance_m": {
            k: float(v.travelled_distance_m) for k, v in metrics.robot_kpi.items()
        },
        "pedestrians_enabled": bool(scene.pedestrians),
        "pedestrian_paths": ped_paths,
        "min_pedestrian_clearance_m": float(state.min_ped_clearance_m)
        if state.min_ped_clearance_m is not None and scene.pedestrians
        else None,
        "robot_robot_collisions": int(state.robot_robot_collisions),
        "robot_ped_violations": int(state.robot_ped_violations),
        "robot_paths": robot_paths,
    }
    return res

