from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from coverage_lab.env.grid_world import GridWorld2D, WorldState
from coverage_lab.metrics import CoverageMetricsLab
from coverage_lab.types import LabScene


@dataclass
class PPOConfig:
    step_scale_m: float = 1.15
    near_thresh_m: float = 0.32

    # Reward shaping tuned for faster learning of coverage objective.
    coverage_scale: float = 160.0
    collision_penalty: float = 3.5
    near_penalty: float = 0.75
    step_penalty: float = 0.01
    success_bonus: float = 45.0
    new_cell_bonus: float = 1.4


class CoveragePedEnv(gym.Env):
    """Gym env for multi-robot coverage with dynamic pedestrians (simple)."""

    metadata = {"render_modes": []}

    def __init__(self, scene: LabScene, seed: int = 0, cfg: PPOConfig | None = None) -> None:
        super().__init__()
        self.scene = scene
        self.seed = int(seed)
        self.cfg = cfg or PPOConfig()

        self.world = GridWorld2D(
            bounds_xy=scene.bounds_xy,
            dt_sec=scene.dt_sec,
            grid_resolution_m=scene.grid_resolution_m,
            obstacles=scene.obstacles,
            rectangles=scene.rectangles,
            pedestrians=scene.pedestrians,
            seed=self.seed,
            ped_safe_m=self.cfg.near_thresh_m,
            max_step_m=0.25,
        )

        # Observation: per-robot (x,y) normalized + nearest 2 peds rel vecs + local visited ratio
        obs_dim = 2 + 4 + 1
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(scene.num_robots, obs_dim), dtype=np.float32)
        # Action: per-robot discrete 5 moves (N,S,E,W,stay)
        self.action_space = gym.spaces.MultiDiscrete([5] * scene.num_robots)

        self._reset_state()

    def _reset_state(self) -> None:
        spawn = self.world.sample_spawn_points(self.scene.num_robots)
        self.state = WorldState(
            robot_xy={f"robot_{i}": spawn[i].copy() for i in range(self.scene.num_robots)},
            robot_target_xy={f"robot_{i}": spawn[i].copy() for i in range(self.scene.num_robots)},
            pedestrians=self.world.pedestrians,
            blocked_moves=0,
            robot_robot_collisions=0,
            robot_ped_violations=0,
            min_ped_clearance_m=None,
        )
        self.metrics = CoverageMetricsLab(bounds_xy=self.scene.bounds_xy, grid_resolution_m=self.scene.grid_resolution_m)
        self.metrics.set_blocked_cells(set(self.world.obstacle_cells()))
        self.metrics.update(self.state.robot_xy)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.seed = int(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        prev_cov = self.metrics.coverage_percent()
        prev_cells = len(self.metrics.visited_cells)

        targets = {}
        step = float(self.scene.grid_resolution_m) * float(self.cfg.step_scale_m)
        for i in range(self.scene.num_robots):
            name = f"robot_{i}"
            p = self.state.robot_xy[name]
            a = int(action[i])
            if a == 0:  # N
                d = np.array([0.0, step])
            elif a == 1:  # S
                d = np.array([0.0, -step])
            elif a == 2:  # E
                d = np.array([step, 0.0])
            elif a == 3:  # W
                d = np.array([-step, 0.0])
            else:
                d = np.zeros(2)
            targets[name] = p + d

        prev = {k: v.copy() for k, v in self.state.robot_xy.items()}
        self.state = self.world.step(self.state, targets)
        for name, p0 in prev.items():
            p1 = self.state.robot_xy[name]
            self.metrics.add_robot_motion(name, float(np.linalg.norm(p1 - p0)))

        self.metrics.update(self.state.robot_xy)
        cov = self.metrics.coverage_percent()
        new_cells = len(self.metrics.visited_cells) - prev_cells

        # reward
        r = 0.0
        r += self.cfg.coverage_scale * (cov - prev_cov)
        r += self.cfg.new_cell_bonus * float(max(new_cells, 0))
        r -= self.cfg.step_penalty
        r -= self.cfg.collision_penalty * float(self.state.robot_robot_collisions)
        r -= self.cfg.near_penalty * float(self.state.robot_ped_violations)

        terminated = cov >= float(self.scene.target_coverage)
        if terminated:
            r += self.cfg.success_bonus

        truncated = self._t >= int(self.scene.max_steps)
        return self._obs(), float(r), bool(terminated), bool(truncated), {
            "coverage": float(cov),
            "new_cells": int(max(new_cells, 0)),
            "blocked_moves": int(self.state.blocked_moves),
            "ped_violations": int(self.state.robot_ped_violations),
        }

    def _obs(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.scene.bounds_xy
        scale_x = max(1e-6, x_max - x_min)
        scale_y = max(1e-6, y_max - y_min)

        peds = [p.position.copy() for p in self.state.pedestrians]
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i in range(self.scene.num_robots):
            name = f"robot_{i}"
            p = self.state.robot_xy[name]
            px = (float(p[0]) - x_min) / scale_x * 2.0 - 1.0
            py = (float(p[1]) - y_min) / scale_y * 2.0 - 1.0
            # nearest 2 pedestrians rel vectors
            rel = []
            for ped in peds:
                rel.append(ped - p)
            rel = sorted(rel, key=lambda v: float(np.linalg.norm(v)))
            rel2 = rel[:2] if rel else [np.zeros(2), np.zeros(2)]
            while len(rel2) < 2:
                rel2.append(np.zeros(2))
            r1 = rel2[0]
            r2 = rel2[1]
            # normalize rel by bounds span
            r1x = float(np.clip(r1[0] / scale_x, -1, 1))
            r1y = float(np.clip(r1[1] / scale_y, -1, 1))
            r2x = float(np.clip(r2[0] / scale_x, -1, 1))
            r2y = float(np.clip(r2[1] / scale_y, -1, 1))

            visited_ratio = float(self.metrics.coverage_percent() * 2.0 - 1.0)
            obs[i, :] = np.array([px, py, r1x, r1y, r2x, r2y, visited_ratio], dtype=np.float32)
        return obs

