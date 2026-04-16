from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

BoundsXY = Tuple[float, float, float, float]


@dataclass
class RobotKpi:
    travelled_distance_m: float = 0.0


@dataclass
class CoverageMetricsLab:
    bounds_xy: BoundsXY
    grid_resolution_m: float
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    visited_cells: set[tuple[int, int]] = field(default_factory=set)
    coverage_history: list[float] = field(default_factory=list)
    distance_history: list[float] = field(default_factory=list)
    robot_kpi: Dict[str, RobotKpi] = field(default_factory=dict)

    def set_blocked_cells(self, blocked: set[tuple[int, int]]) -> None:
        self.blocked_cells = set(blocked)

    def update(self, robot_xy: Dict[str, np.ndarray]) -> None:
        for name, p in robot_xy.items():
            cell = self._to_cell(p)
            if cell not in self.blocked_cells:
                self.visited_cells.add(cell)
            self.robot_kpi.setdefault(name, RobotKpi())
        self.coverage_history.append(self.coverage_percent())
        self.distance_history.append(self.total_distance())

    def add_robot_motion(self, name: str, dist_m: float) -> None:
        self.robot_kpi.setdefault(name, RobotKpi())
        self.robot_kpi[name].travelled_distance_m += float(dist_m)

    def total_distance(self) -> float:
        return float(sum(k.travelled_distance_m for k in self.robot_kpi.values()))

    def coverage_percent(self) -> float:
        total = self._total_free_cells()
        if total <= 0:
            return 0.0
        return len(self.visited_cells) / total

    def efficiency(self) -> float:
        d = self.total_distance()
        return 0.0 if d <= 1e-9 else self.coverage_percent() / d

    def load_balance_cv(self) -> float:
        dists = [k.travelled_distance_m for k in self.robot_kpi.values()]
        if not dists:
            return 0.0
        arr = np.asarray(dists, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        return 0.0 if mean <= 1e-9 else float(std / mean)

    def time_to_coverage_sec(self, dt_sec: float, target_cov: float) -> float | None:
        for i, c in enumerate(self.coverage_history):
            if c >= target_cov:
                return float(i) * float(dt_sec)
        return None

    def _to_cell(self, p: np.ndarray) -> tuple[int, int]:
        x_min, _, y_min, _ = self.bounds_xy
        return (
            int((float(p[0]) - x_min) / self.grid_resolution_m),
            int((float(p[1]) - y_min) / self.grid_resolution_m),
        )

    def _total_free_cells(self) -> int:
        x_min, x_max, y_min, y_max = self.bounds_xy
        nx = int(np.ceil((x_max - x_min) / self.grid_resolution_m))
        ny = int(np.ceil((y_max - y_min) / self.grid_resolution_m))
        total = max(nx * ny, 1)
        free = total - len(self.blocked_cells)
        return max(free, 1)

