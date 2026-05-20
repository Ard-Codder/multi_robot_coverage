from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

Cell = Tuple[int, int]


@dataclass(frozen=True)
class GoalDatasetConfig:
    window: int = 31
    stride: int = 4
    max_samples_per_run: int = 30_000
    seed: int = 0
    label_mode: str = "teacher_frontier"
    rollout_k: int = 16


def _to_cell(p: np.ndarray, *, x_min: float, y_min: float, res: float) -> Cell:
    return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)


def _inside(c: Cell, nx: int, ny: int) -> bool:
    return 0 <= c[0] < nx and 0 <= c[1] < ny


def _frontier_cells(*, visited: set[Cell], blocked: set[Cell], nx: int, ny: int) -> set[Cell]:
    frontiers: set[Cell] = set()
    for cx, cy in visited:
        for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if _inside(nb, nx, ny) and nb not in visited and nb not in blocked:
                frontiers.add(nb)
    return frontiers


def _choose_goal(
    *,
    robot_cell: Cell,
    next_cell: Cell,
    visited: set[Cell],
    blocked: set[Cell],
    nx: int,
    ny: int,
    radius: int,
) -> Cell:
    """Choose a supervised goal inside the local window.

    Prefer the teacher's future cell if it is meaningful and inside the window.
    Otherwise use the nearest frontier cell in the local window. This keeps the
    target connected to the teacher trajectory while avoiding degenerate stop
    labels that broke the step-level classifier.
    """
    dx = next_cell[0] - robot_cell[0]
    dy = next_cell[1] - robot_cell[1]
    if abs(dx) <= radius and abs(dy) <= radius and next_cell not in blocked and _inside(next_cell, nx, ny):
        if next_cell != robot_cell:
            return next_cell

    frontiers = _frontier_cells(visited=visited, blocked=blocked, nx=nx, ny=ny)
    local = [
        c
        for c in frontiers
        if abs(c[0] - robot_cell[0]) <= radius and abs(c[1] - robot_cell[1]) <= radius
    ]
    if local:
        return min(local, key=lambda c: abs(c[0] - robot_cell[0]) + abs(c[1] - robot_cell[1]))

    # Last-resort non-stop target: nearest free unvisited cell in the window.
    candidates: list[Cell] = []
    for dx0 in range(-radius, radius + 1):
        for dy0 in range(-radius, radius + 1):
            c = (robot_cell[0] + dx0, robot_cell[1] + dy0)
            if _inside(c, nx, ny) and c not in blocked and c not in visited:
                candidates.append(c)
    if candidates:
        return min(candidates, key=lambda c: abs(c[0] - robot_cell[0]) + abs(c[1] - robot_cell[1]))
    return robot_cell


def _best_of_k_goal(
    *,
    robot_cell: Cell,
    visited: set[Cell],
    blocked: set[Cell],
    frontier: set[Cell],
    nx: int,
    ny: int,
    radius: int,
    k: int,
) -> Cell:
    local_frontier = [
        c
        for c in frontier
        if abs(c[0] - robot_cell[0]) <= radius and abs(c[1] - robot_cell[1]) <= radius
    ]
    candidates = local_frontier
    if not candidates:
        candidates = [
            (robot_cell[0] + dx, robot_cell[1] + dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if _inside((robot_cell[0] + dx, robot_cell[1] + dy), nx, ny)
            and (robot_cell[0] + dx, robot_cell[1] + dy) not in blocked
            and (robot_cell[0] + dx, robot_cell[1] + dy) not in visited
        ]
    if not candidates:
        return robot_cell
    candidates = sorted(candidates, key=lambda c: abs(c[0] - robot_cell[0]) + abs(c[1] - robot_cell[1]))[: max(1, int(k))]

    def score(c: Cell) -> float:
        new_cells = 0.0 if c in visited else 1.0
        for nb in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if _inside(nb, nx, ny) and nb not in visited and nb not in blocked:
                new_cells += 0.35
        dist = abs(c[0] - robot_cell[0]) + abs(c[1] - robot_cell[1])
        return new_cells - 0.04 * float(dist)

    return max(candidates, key=score)


def _make_obs(
    *,
    robot_cell: Cell,
    visited: set[Cell],
    blocked: set[Cell],
    other_robots: set[Cell],
    frontier: set[Cell],
    nx: int,
    ny: int,
    window: int,
) -> np.ndarray:
    radius = window // 2
    obs = np.zeros((6, window, window), dtype=np.float32)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cell = (robot_cell[0] + dx, robot_cell[1] + dy)
            if not _inside(cell, nx, ny):
                continue
            px, py = dy + radius, dx + radius
            if cell in visited:
                obs[0, px, py] = 1.0
            if cell in blocked:
                obs[1, px, py] = 1.0
            if cell not in blocked and cell not in visited:
                obs[2, px, py] = 1.0
            if dx == 0 and dy == 0:
                obs[3, px, py] = 1.0
            if cell in other_robots:
                obs[4, px, py] = 1.0
            if cell in frontier:
                obs[5, px, py] = 1.0
    return obs


def _iter_samples_from_run(path: Path, cfg: GoalDatasetConfig) -> Iterable[tuple[np.ndarray, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    x_min, x_max, y_min, y_max = [float(v) for v in data["bounds_xy"]]
    res = float(data["grid_resolution_m"])
    nx = max(1, int(np.ceil((x_max - x_min) / res)))
    ny = max(1, int(np.ceil((y_max - y_min) / res)))
    blocked = {tuple(c) for c in data.get("obstacle_cells", [])}
    paths: Dict[str, List[List[float]]] = data.get("robot_paths", {})
    arrays = {
        name: np.asarray(points, dtype=float)
        for name, points in paths.items()
        if points and len(points) >= 2
    }
    if not arrays:
        return

    window = int(cfg.window)
    radius = window // 2
    max_len = max(len(arr) for arr in arrays.values())
    visited: set[Cell] = set()
    made = 0
    for i in range(max_len - 1):
        if i % max(1, int(cfg.stride)) != 0:
            for arr in arrays.values():
                if i < len(arr):
                    visited.add(_to_cell(arr[i], x_min=x_min, y_min=y_min, res=res))
            continue
        current_cells = {
            name: _to_cell(arr[i], x_min=x_min, y_min=y_min, res=res)
            for name, arr in arrays.items()
            if i < len(arr)
        }
        visited.update(current_cells.values())
        frontier = _frontier_cells(visited=visited, blocked=blocked, nx=nx, ny=ny)
        for name, arr in arrays.items():
            if i >= len(arr) - 1:
                continue
            robot_cell = _to_cell(arr[i], x_min=x_min, y_min=y_min, res=res)
            next_cell = _to_cell(arr[min(i + cfg.stride, len(arr) - 1)], x_min=x_min, y_min=y_min, res=res)
            if cfg.label_mode == "best_of_k":
                goal = _best_of_k_goal(
                    robot_cell=robot_cell,
                    visited=visited,
                    blocked=blocked,
                    frontier=frontier,
                    nx=nx,
                    ny=ny,
                    radius=radius,
                    k=int(cfg.rollout_k),
                )
            else:
                goal = _choose_goal(
                    robot_cell=robot_cell,
                    next_cell=next_cell,
                    visited=visited,
                    blocked=blocked,
                    nx=nx,
                    ny=ny,
                    radius=radius,
                )
            other = {c for robot, c in current_cells.items() if robot != name}
            obs = _make_obs(
                robot_cell=robot_cell,
                visited=visited,
                blocked=blocked,
                other_robots=other,
                frontier=frontier,
                nx=nx,
                ny=ny,
                window=window,
            )
            rel_x = int(np.clip(goal[0] - robot_cell[0], -radius, radius))
            rel_y = int(np.clip(goal[1] - robot_cell[1], -radius, radius))
            label = (rel_y + radius) * window + (rel_x + radius)
            yield obs, int(label)
            made += 1
            if cfg.max_samples_per_run and made >= int(cfg.max_samples_per_run):
                return


def build_goal_dataset_from_runs(
    json_paths: list[Path],
    *,
    window: int = 31,
    stride: int = 4,
    max_samples_per_run: int = 30_000,
    label_mode: str = "teacher_frontier",
    rollout_k: int = 16,
    max_samples: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = GoalDatasetConfig(
        window=window,
        stride=stride,
        max_samples_per_run=max_samples_per_run,
        seed=seed,
        label_mode=label_mode,
        rollout_k=rollout_k,
    )
    samples: list[tuple[np.ndarray, int]] = []
    for path in json_paths:
        samples.extend(_iter_samples_from_run(path, cfg))
        if max_samples and len(samples) >= max_samples:
            break
    if max_samples and len(samples) > max_samples:
        rng = random.Random(int(seed))
        samples = rng.sample(samples, int(max_samples))
    if not samples:
        return np.zeros((0, 6, window, window), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack([s[0] for s in samples], axis=0)
    y = np.asarray([s[1] for s in samples], dtype=np.int64)
    return X, y
