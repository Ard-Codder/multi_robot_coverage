from __future__ import annotations

import numpy as np


class ProportionalController:
    def __init__(self, kp_linear: float = 1.2, max_step_m: float = 0.25) -> None:
        self.kp_linear = kp_linear
        self.max_step_m = max_step_m

    def compute_next_position(self, current_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        error = target_xy - current_xy
        norm = float(np.linalg.norm(error))
        if norm < 1e-6:
            return current_xy.copy()
        velocity = self.kp_linear * error
        vel_norm = float(np.linalg.norm(velocity))
        if vel_norm < 1e-6:
            return current_xy.copy()
        step = min(self.max_step_m, vel_norm)
        return current_xy + step * velocity / vel_norm

