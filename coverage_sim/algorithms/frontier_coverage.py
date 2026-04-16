from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class FrontierCoverage:
    bounds_xy: Bounds
    grid_resolution_m: float = 1.0
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    seed: int | None = 7

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_targets(self, robot_states: Dict[str, RobotState]) -> Dict[str, np.ndarray]:
        for state in robot_states.values():
            self.visited.add(self._to_cell(state.position))

        frontiers = self._frontier_cells()
        targets: Dict[str, np.ndarray] = {}
        assigned: Set[Tuple[int, int]] = set()
        for name in sorted(robot_states.keys()):
            state = robot_states[name]
            if not frontiers:
                targets[name] = state.target.copy()
                continue
            candidates = [c for c in frontiers if c not in assigned]
            if not candidates:
                candidates = list(frontiers)
            pos = state.position
            best_cell = min(
                candidates,
                key=lambda c: float(np.linalg.norm(self._to_world(c) - pos)),
            )
            assigned.add(best_cell)
            targets[name] = self._to_world(best_cell)
        return targets

    def _frontier_cells(self) -> list[Tuple[int, int]]:
        cells = []
        for cell in self.visited:
            for nb in self._neighbors(cell):
                if nb not in self.visited and self._in_bounds_cell(nb):
                    cells.append(nb)
        return cells

    def _neighbors(self, cell: Tuple[int, int]) -> list[Tuple[int, int]]:
        cx, cy = cell
        return [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]

    def _to_cell(self, point: np.ndarray) -> Tuple[int, int]:
        x_min, _, y_min, _ = self.bounds_xy
        return (
            int((point[0] - x_min) / self.grid_resolution_m),
            int((point[1] - y_min) / self.grid_resolution_m),
        )

    def _to_world(self, cell: Tuple[int, int]) -> np.ndarray:
        x_min, _, y_min, _ = self.bounds_xy
        return np.array(
            [
                x_min + (cell[0] + 0.5) * self.grid_resolution_m,
                y_min + (cell[1] + 0.5) * self.grid_resolution_m,
            ],
            dtype=float,
        )

    def _in_bounds_cell(self, cell: Tuple[int, int]) -> bool:
        x_min, x_max, y_min, y_max = self.bounds_xy
        px = x_min + (cell[0] + 0.5) * self.grid_resolution_m
        py = y_min + (cell[1] + 0.5) * self.grid_resolution_m
        return x_min <= px <= x_max and y_min <= py <= y_max

