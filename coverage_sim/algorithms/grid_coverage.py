from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class GridCoverage:
    bounds_xy: Bounds
    cell_size_m: float = 2.0
    reach_threshold_m: float = 0.4

    def __post_init__(self) -> None:
        self._paths: Dict[str, List[np.ndarray]] = {}
        self._indices: Dict[str, int] = {}

    def choose_targets(self, robot_states: Dict[str, RobotState]) -> Dict[str, np.ndarray]:
        if not self._paths:
            self._init_paths(list(robot_states.keys()))

        targets: Dict[str, np.ndarray] = {}
        for name, state in robot_states.items():
            path = self._paths[name]
            idx = self._indices[name]
            if idx >= len(path):
                targets[name] = state.target.copy()
                continue
            current_target = path[idx]
            if float(np.linalg.norm(current_target - state.position)) <= self.reach_threshold_m:
                idx = min(idx + 1, len(path) - 1)
                self._indices[name] = idx
                current_target = path[idx]
            targets[name] = current_target.copy()
        return targets

    def _init_paths(self, names: List[str]) -> None:
        x_min, x_max, y_min, y_max = self.bounds_xy
        xs = np.arange(x_min, x_max + 1e-9, self.cell_size_m)
        ys = np.arange(y_min, y_max + 1e-9, self.cell_size_m)
        points = [np.array([x, y], dtype=float) for y in ys for x in xs]
        chunks = [points[i:: len(names)] for i in range(len(names))]
        for name, chunk in zip(names, chunks):
            self._paths[name] = chunk if chunk else [np.array([x_min, y_min], dtype=float)]
            self._indices[name] = 0

