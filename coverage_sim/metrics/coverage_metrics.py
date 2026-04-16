from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class CoverageMetrics:
    bounds_xy: Bounds
    grid_resolution_m: float = 0.5
    visited_cells: set[tuple[int, int]] = field(default_factory=set)
    coverage_history: list[float] = field(default_factory=list)
    distance_history: list[float] = field(default_factory=list)
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)

    def set_blocked_cells(self, blocked_cells: set[tuple[int, int]]) -> None:
        self.blocked_cells = set(blocked_cells)

    def update(self, robot_states: Dict[str, RobotState]) -> None:
        for state in robot_states.values():
            cell = self._to_cell(state.position)
            if cell not in self.blocked_cells:
                self.visited_cells.add(cell)
        self.coverage_history.append(self.coverage_percent())
        self.distance_history.append(self.total_distance(robot_states))

    def coverage_percent(self) -> float:
        total = self._total_cells()
        if total <= 0:
            return 0.0
        return len(self.visited_cells) / total

    def total_distance(self, robot_states: Dict[str, RobotState]) -> float:
        return float(sum(state.travelled_distance for state in robot_states.values()))

    def efficiency(self, robot_states: Dict[str, RobotState]) -> float:
        dist = self.total_distance(robot_states)
        if dist <= 1e-9:
            return 0.0
        return self.coverage_percent() / dist

    def load_balance_stats(self, robot_states: Dict[str, RobotState]) -> dict:
        dists = [float(st.travelled_distance) for st in robot_states.values()]
        if not dists:
            return {
                "per_robot_distance_m": {},
                "distance_mean_m": 0.0,
                "distance_std_m": 0.0,
                "load_balance_cv": 0.0,
            }
        arr = np.asarray(dists, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        cv = std / mean if mean > 1e-9 else 0.0
        return {
            "per_robot_distance_m": {
                name: float(st.travelled_distance) for name, st in robot_states.items()
            },
            "distance_mean_m": mean,
            "distance_std_m": std,
            "load_balance_cv": cv,
        }

    def to_dict(
        self,
        robot_states: Dict[str, RobotState],
        dt_sec: float,
        *,
        target_coverage_for_ttc: float = 0.99,
    ) -> dict:
        ttc = None
        for idx, cov in enumerate(self.coverage_history):
            if cov >= target_coverage_for_ttc:
                ttc = idx * dt_sec
                break
        lb = self.load_balance_stats(robot_states)
        return {
            "coverage_percent": self.coverage_percent(),
            "time_to_coverage_sec": ttc,
            "distance_travelled_m": self.total_distance(robot_states),
            "efficiency": self.efficiency(robot_states),
            **lb,
            "coverage_history": self.coverage_history,
            "distance_history": self.distance_history,
            "visited_cells": [[c[0], c[1]] for c in sorted(self.visited_cells)],
            "bounds_xy": list(self.bounds_xy),
            "grid_resolution_m": self.grid_resolution_m,
        }

    def _to_cell(self, point: np.ndarray) -> tuple[int, int]:
        x_min, _, y_min, _ = self.bounds_xy
        return (
            int((point[0] - x_min) / self.grid_resolution_m),
            int((point[1] - y_min) / self.grid_resolution_m),
        )

    def _total_cells(self) -> int:
        x_min, x_max, y_min, y_max = self.bounds_xy
        nx = int(np.ceil((x_max - x_min) / self.grid_resolution_m))
        ny = int(np.ceil((y_max - y_min) / self.grid_resolution_m))
        total = max(nx * ny, 1)
        free = total - len(self.blocked_cells)
        return max(free, 1)

