from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class ZonalRandomWalkCoverage:
    """
    Простая мультиагентная зональная стратегия.
    Пространство делится по ближайшему seed (аналог Voronoi на seed'ах),
    после чего каждый робот ходит случайно только внутри своей зоны.
    """

    bounds_xy: Bounds
    reach_threshold_m: float = 0.5
    seed: int | None = 12
    zone_grid_step_m: float = 1.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._zones_built = False
        self._robot_cells: Dict[str, np.ndarray] = {}

    def choose_targets(self, robot_states: Dict[str, RobotState]) -> Dict[str, np.ndarray]:
        if not self._zones_built:
            self._build_zones(robot_states)

        targets: Dict[str, np.ndarray] = {}
        for name, state in robot_states.items():
            if float(np.linalg.norm(state.target - state.position)) <= self.reach_threshold_m:
                targets[name] = self._sample_zone_point(name)
            else:
                targets[name] = state.target.copy()
        return targets

    def _build_zones(self, robot_states: Dict[str, RobotState]) -> None:
        names = list(robot_states.keys())
        seeds = {n: robot_states[n].position.copy() for n in names}
        x_min, x_max, y_min, y_max = self.bounds_xy
        xs = np.arange(x_min, x_max + 1e-9, self.zone_grid_step_m)
        ys = np.arange(y_min, y_max + 1e-9, self.zone_grid_step_m)
        cells: Dict[str, list[np.ndarray]] = {n: [] for n in names}

        for x in xs:
            for y in ys:
                p = np.array([x, y], dtype=float)
                owner = min(names, key=lambda n: float(np.linalg.norm(p - seeds[n])))
                cells[owner].append(p)

        for n in names:
            if not cells[n]:
                cells[n] = [seeds[n]]
            self._robot_cells[n] = np.array(cells[n], dtype=float)
        self._zones_built = True

    def _sample_zone_point(self, robot_name: str) -> np.ndarray:
        arr = self._robot_cells[robot_name]
        idx = int(self.rng.integers(0, arr.shape[0]))
        return arr[idx].copy()

