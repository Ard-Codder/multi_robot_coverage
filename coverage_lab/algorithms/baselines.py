from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from coverage_lab.env.grid_world import GridWorld2D
from coverage_lab.types import LabScene

Cell = Tuple[int, int]
STAGNATION_EPS = 1e-6
STAGNATION_STEPS = 14


def _grid_meta(scene: LabScene) -> Tuple[float, float, int, int]:
    x_min, x_max, y_min, y_max = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    nx = max(1, int(np.ceil((x_max - x_min) / res)))
    ny = max(1, int(np.ceil((y_max - y_min) / res)))
    return x_min, y_min, nx, ny


def _to_cell(scene: LabScene, p: np.ndarray) -> Cell:
    x_min, y_min, _, _ = _grid_meta(scene)
    res = float(scene.grid_resolution_m)
    return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)


def _cell_center(scene: LabScene, c: Cell) -> np.ndarray:
    x_min, y_min, _, _ = _grid_meta(scene)
    res = float(scene.grid_resolution_m)
    return np.array([x_min + (c[0] + 0.5) * res, y_min + (c[1] + 0.5) * res], dtype=float)


def _nearest_unvisited(
    *,
    scene: LabScene,
    blocked: Set[Cell],
    visited: Set[Cell],
    start: Cell,
    forbidden: Set[Cell] | None = None,
    candidates: Set[Cell] | None = None,
) -> Cell | None:
    _, _, nx, ny = _grid_meta(scene)
    avoid = forbidden or set()
    best: Cell | None = None
    bestd = 10**9
    if candidates is None:
        for cx in range(nx):
            for cy in range(ny):
                c = (cx, cy)
                if c in blocked or c in visited or c in avoid:
                    continue
                d = abs(c[0] - start[0]) + abs(c[1] - start[1])
                if d < bestd:
                    bestd = d
                    best = c
    else:
        for c in candidates:
            if c in blocked or c in visited or c in avoid:
                continue
            d = abs(c[0] - start[0]) + abs(c[1] - start[1])
            if d < bestd:
                bestd = d
                best = c
    return best


class BaseAlgo:
    def choose_targets(
        self,
        robot_xy: Dict[str, np.ndarray],
        *,
        scene: LabScene,
        world: GridWorld2D,
        metrics=None,
    ):
        raise NotImplementedError


@dataclass
class BaselineRandomWalk(BaseAlgo):
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._ang = {}

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        out = {}
        for name, p in robot_xy.items():
            a = float(self._ang.get(name, self._rng.uniform(0, 2 * math.pi)))
            a += float(self._rng.normal(0.0, 0.55))
            self._ang[name] = a
            step = max(world.max_step_m * 2.0, scene.grid_resolution_m * 0.9)
            t = p + np.array([math.cos(a), math.sin(a)], dtype=float) * step
            out[name] = t
        return out


class BaselineGrid(BaseAlgo):
    """Very simple lawnmower on bounding box; phase shift per robot."""

    def __init__(self) -> None:
        self._dir = {}
        self._last_cov = 0.0
        self._stagnation = 0

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        x_min, x_max, y_min, y_max = scene.bounds_xy
        dx = scene.grid_resolution_m * 2.0
        blocked = set(metrics.blocked_cells) if metrics is not None and hasattr(metrics, "blocked_cells") else set()
        visited = set(metrics.visited_cells) if metrics is not None and hasattr(metrics, "visited_cells") else set()
        if metrics is not None and hasattr(metrics, "coverage_percent"):
            cov = float(metrics.coverage_percent())
            if cov <= self._last_cov + STAGNATION_EPS:
                self._stagnation += 1
            else:
                self._stagnation = 0
            self._last_cov = cov

        force_escape = metrics is not None and self._stagnation >= STAGNATION_STEPS
        reserved: Set[Cell] = set()
        out = {}
        for name, p in sorted(robot_xy.items()):
            d = int(self._dir.get(name, 1))
            y = float(np.clip(p[1] + d * dx, y_min, y_max))
            x = float(p[0])
            if (d > 0 and y >= y_max - 1e-6) or (d < 0 and y <= y_min + 1e-6):
                # switch direction and advance x
                d = -d
                self._dir[name] = d
                x = float(np.clip(p[0] + dx, x_min, x_max))
            target = np.array([x, y], dtype=float)
            tcell = _to_cell(scene, target)
            if force_escape or tcell in reserved or tcell in blocked:
                start = _to_cell(scene, p)
                alt = _nearest_unvisited(
                    scene=scene,
                    blocked=blocked,
                    visited=visited,
                    start=start,
                    forbidden=reserved,
                )
                if alt is not None:
                    target = _cell_center(scene, alt)
                    tcell = alt
            reserved.add(tcell)
            out[name] = target
        return out


class BaselineVoronoi(BaseAlgo):
    """Voronoi-ish (coverage-driven): partition free cells by nearest robot, go to nearest unvisited in own region."""

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        # fallback if no metrics
        if metrics is None:
            return {k: v.copy() for k, v in robot_xy.items()}

        x_min, x_max, y_min, y_max = scene.bounds_xy
        res = float(scene.grid_resolution_m)
        nx = max(1, int(np.ceil((x_max - x_min) / res)))
        ny = max(1, int(np.ceil((y_max - y_min) / res)))

        blocked = set(metrics.blocked_cells) if hasattr(metrics, "blocked_cells") else set(world.obstacle_cells())
        visited = set(metrics.visited_cells) if hasattr(metrics, "visited_cells") else set()
        names = sorted(robot_xy.keys())
        start_cells = {n: (int((robot_xy[n][0] - x_min) / res), int((robot_xy[n][1] - y_min) / res)) for n in names}

        # assign each free unvisited cell to nearest robot (L1)
        regions = {n: [] for n in names}
        for cx in range(nx):
            for cy in range(ny):
                c = (cx, cy)
                if c in blocked or c in visited:
                    continue
                best = None
                bestd = 1e9
                for n in names:
                    sx, sy = start_cells[n]
                    d = abs(cx - sx) + abs(cy - sy)
                    if d < bestd:
                        bestd = d
                        best = n
                regions[str(best)].append(c)

        def cell_center(c):
            return np.array([x_min + (c[0] + 0.5) * res, y_min + (c[1] + 0.5) * res], dtype=float)

        reserved: Set[Cell] = set()
        out = {}
        for n in names:
            candidates: List[Cell] = list(regions.get(n) or [])
            sx, sy = start_cells[n]

            # Prefer nearest in own region, but avoid duplicating exact same target across robots.
            avail = [c for c in candidates if c not in reserved]
            if avail:
                tgt = min(avail, key=lambda c: abs(c[0] - sx) + abs(c[1] - sy))
                reserved.add(tgt)
                out[n] = cell_center(tgt)
                continue

            # Fallback: pick a global nearest unvisited cell if region is empty or fully reserved.
            start = start_cells[n]
            alt = _nearest_unvisited(
                scene=scene,
                blocked=blocked,
                visited=visited,
                start=start,
                forbidden=reserved,
            )
            if alt is not None:
                reserved.add(alt)
                out[n] = cell_center(alt)
            else:
                out[n] = robot_xy[n].copy()
        return out


class BaselineFrontier(BaseAlgo):
    """Frontier-like (coverage-driven): go to nearest unvisited cell (greedy)."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._last_cov = 0.0
        self._stagnation = 0

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        if metrics is None:
            return {k: v.copy() for k, v in robot_xy.items()}

        x_min, y_min, nx, ny = _grid_meta(scene)
        res = float(scene.grid_resolution_m)
        blocked = set(metrics.blocked_cells) if hasattr(metrics, "blocked_cells") else set(world.obstacle_cells())
        visited = set(metrics.visited_cells) if hasattr(metrics, "visited_cells") else set()
        cov = float(metrics.coverage_percent()) if hasattr(metrics, "coverage_percent") else 0.0
        if cov <= self._last_cov + STAGNATION_EPS:
            self._stagnation += 1
        else:
            self._stagnation = 0
        self._last_cov = cov

        # precompute list of unvisited cells (can be large, but ok for our sizes)
        unvisited = [(cx, cy) for cx in range(nx) for cy in range(ny) if (cx, cy) not in blocked and (cx, cy) not in visited]
        reserved: Set[Cell] = set()

        out = {}
        for name, p in sorted(robot_xy.items()):
            if not unvisited:
                out[name] = p.copy()
                continue
            sx = int((float(p[0]) - x_min) / res)
            sy = int((float(p[1]) - y_min) / res)
            avail = [c for c in unvisited if c not in reserved]
            if not avail:
                avail = unvisited
            if self._stagnation >= STAGNATION_STEPS:
                # During stagnation, prefer farther frontier cells to avoid local loops.
                tgt = max(avail, key=lambda c: abs(c[0] - sx) + abs(c[1] - sy))
            else:
                tgt = min(avail, key=lambda c: abs(c[0] - sx) + abs(c[1] - sy))
            reserved.add(tgt)
            # small noise to break symmetry
            noise = self._rng.normal(0.0, 0.05, size=(2,))
            out[name] = _cell_center(scene, tgt) + noise
        return out

