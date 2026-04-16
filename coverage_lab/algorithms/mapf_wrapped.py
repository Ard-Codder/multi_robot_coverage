from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from coverage_lab.env.grid_world import GridWorld2D
from coverage_lab.mapf.cbs import cbs_solve
from coverage_lab.types import LabScene


Cell = Tuple[int, int]


def _grid_dims(scene: LabScene) -> Tuple[int, int]:
    x_min, x_max, y_min, y_max = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    nx = max(1, int(np.ceil((x_max - x_min) / res)))
    ny = max(1, int(np.ceil((y_max - y_min) / res)))
    return nx, ny


def _to_cell(scene: LabScene, p: np.ndarray) -> Cell:
    x_min, _, y_min, _ = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    return int((float(p[0]) - x_min) / res), int((float(p[1]) - y_min) / res)


def _cell_center(scene: LabScene, c: Cell) -> np.ndarray:
    x_min, _, y_min, _ = scene.bounds_xy
    res = float(scene.grid_resolution_m)
    return np.array([x_min + (c[0] + 0.5) * res, y_min + (c[1] + 0.5) * res], dtype=float)


@dataclass
class CBSDeconflictWrapper:
    """Wraps an underlying target-choosing algorithm with CBS path deconfliction on grid."""

    inner: any
    horizon: int = 32
    _plans: Dict[str, List[Cell]] = None  # type: ignore[assignment]
    _idx: Dict[str, int] = None  # type: ignore[assignment]
    _disabled: bool = False

    def __post_init__(self) -> None:
        self._plans = {}
        self._idx = {}
        self._disabled = False

    def choose_targets(self, robot_xy: Dict[str, np.ndarray], *, scene: LabScene, world: GridWorld2D, metrics=None):
        if self._disabled:
            return self.inner.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)

        # if we have remaining plan steps, follow them
        if self._plans and all(self._idx.get(a, 0) < len(p) for a, p in self._plans.items()):
            out = {}
            for a, path in self._plans.items():
                i = int(self._idx.get(a, 0))
                c = path[min(i, len(path) - 1)]
                out[a] = _cell_center(scene, c)
                self._idx[a] = i + 1
            return out

        # otherwise compute new CBS plan from current cells to inner goals
        inner_targets = self.inner.choose_targets(robot_xy, scene=scene, world=world, metrics=metrics)
        starts = {a: _to_cell(scene, p) for a, p in robot_xy.items()}
        goals = {a: _to_cell(scene, inner_targets[a]) for a in robot_xy.keys()}
        blocked = set(world.obstacle_cells())
        plan = cbs_solve(
            starts=starts,
            goals=goals,
            grid_size=_grid_dims(scene),
            blocked=blocked,
            max_t=self.horizon,
            max_nodes=1800,
            max_sec=1.5,
        )
        if plan is None:
            # fallback: just use inner
            self._disabled = True
            return inner_targets

        self._plans = plan.paths
        self._idx = {a: 1 for a in plan.paths.keys()}  # 0 is current
        return {a: _cell_center(scene, path[1] if len(path) > 1 else path[0]) for a, path in plan.paths.items()}

