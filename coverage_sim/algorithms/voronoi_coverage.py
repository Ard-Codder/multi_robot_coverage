from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from coverage_sim.robots.robot_state import RobotState


Bounds = Tuple[float, float, float, float]


@dataclass
class VoronoiCoverage:
    bounds_xy: Bounds
    step_scale_m: float = 1.0

    def choose_targets(self, robot_states: Dict[str, RobotState]) -> Dict[str, np.ndarray]:
        positions = {name: st.position.copy() for name, st in robot_states.items()}
        targets: Dict[str, np.ndarray] = {}

        for name, state in robot_states.items():
            repulsion = np.zeros(2, dtype=float)
            for other_name, other_pos in positions.items():
                if other_name == name:
                    continue
                diff = state.position - other_pos
                dist = float(np.linalg.norm(diff)) + 1e-6
                repulsion += diff / (dist * dist)

            if float(np.linalg.norm(repulsion)) < 1e-6:
                targets[name] = state.target.copy()
                continue

            direction = repulsion / (float(np.linalg.norm(repulsion)) + 1e-6)
            candidate = state.position + self.step_scale_m * direction
            targets[name] = self._clip(candidate)

        return targets

    def _clip(self, point: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.bounds_xy
        return np.array(
            [
                np.clip(point[0], x_min, x_max),
                np.clip(point[1], y_min, y_max),
            ],
            dtype=float,
        )

