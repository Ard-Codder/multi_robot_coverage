"""
Gymnasium-среда: мультиагентное покрытие с «умными» пешеходами (kinematic fallback).
Награда: прирост покрытия, штраф за шаг, штраф за близость/коллизию с пешеходами.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e

from coverage_sim.env.isaac_env import IsaacEnvironment
from coverage_sim.metrics.coverage_metrics import CoverageMetrics
from coverage_sim.robots.robot_state import RobotState
from coverage_sim.simulation.spawn_robots import create_spawn_points


def _action_to_delta(idx: int) -> Tuple[int, int]:
    dx = idx % 3 - 1
    dy = idx // 3 - 1
    return dx, dy


class CoveragePedRLEnv(gym.Env):
    """
    Действие: MultiDiscrete([9]*n_robots) — смещение цели относительно текущей позиции робота
    (индекс 0..8 → сетка 3×3 для (dx,dy) ∈ {-1,0,1}²).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: Optional[Path] = None,
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        cfg_path = config_path or (root / "coverage_sim" / "configs" / "world.yaml")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        robots_raw = yaml.safe_load((root / "coverage_sim" / "configs" / "robots.yaml").read_text(encoding="utf-8"))
        self._world_cfg = raw
        rt = raw.get("rl_training") or {}
        if overrides:
            rt = {**rt, **overrides}

        w = rt.get("bounds_xy") or raw["world"]["bounds_xy"]
        self._bounds = (
            float(w["x_min"]),
            float(w["x_max"]),
            float(w["y_min"]),
            float(w["y_max"]),
        )
        self._dt = float(raw["simulation"]["dt_sec"])
        self._grid_res = float(rt.get("grid_resolution_m", 1.0))
        self._target_cov = float(rt.get("target_coverage", 0.88))
        self._max_episode_steps = int(rt.get("max_episode_steps", 500))
        self._step_scale = float(rt.get("step_scale_m", 1.35))
        self._n_robots = int(rt.get("num_robots", robots_raw["robots"]["count"]))

        rw = rt.get("reward") or {}
        self._r_cov_scale = float(rw.get("coverage_scale", 60.0))
        self._r_collision = float(rw.get("collision_penalty", 10.0))
        self._r_near = float(rw.get("near_penalty", 2.0))
        self._near_thresh = float(rw.get("near_thresh_m", 0.3))
        self._r_step = float(rw.get("step_penalty", 0.03))
        self._r_success = float(rw.get("success_bonus", 20.0))
        self._r_new_cell = float(rw.get("new_cell_bonus", 0.35))
        self._spawn_jitter_m = float(rw.get("spawn_jitter_m", 0.5))

        obs = rt.get("fallback_obstacles") or []
        self._fallback_obs: List[Tuple[float, float, float]] = [
            (float(o["x"]), float(o["y"]), float(o["r"])) for o in obs
        ]

        self._ped_cfg = rt.get("pedestrians")

        xspan = self._bounds[1] - self._bounds[0]
        yspan = self._bounds[3] - self._bounds[2]
        self._nx = max(1, int(math.ceil(xspan / self._grid_res)))
        self._ny = max(1, int(math.ceil(yspan / self._grid_res)))
        obs_dim = self._nx * self._ny + 2 * self._n_robots + 2 * 8 + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9] * self._n_robots)

        self._env: Optional[IsaacEnvironment] = None
        self._metrics: Optional[CoverageMetrics] = None
        self._robot_states: Dict[str, RobotState] = {}
        self._step_count = 0
        self._prev_cov = 0.0
        self._rng = np.random.default_rng(0)

    def _spawn_dict(self) -> Dict[str, Tuple[float, float]]:
        assert self._env is not None
        pts = self._env.suggest_spawn_points(self._n_robots, self._bounds)
        if pts is None:
            pts = create_spawn_points(self._n_robots, self._bounds)
        out: Dict[str, Tuple[float, float]] = {}
        j = max(0.0, self._spawn_jitter_m)
        for i in range(self._n_robots):
            name = f"robot_{i}"
            x, y = float(pts[i][0]), float(pts[i][1])
            if j > 0:
                x += float(self._rng.uniform(-j, j))
                y += float(self._rng.uniform(-j, j))
            x_min, x_max, y_min, y_max = self._bounds
            x = float(np.clip(x, x_min + 0.2, x_max - 0.2))
            y = float(np.clip(y, y_min + 0.2, y_max - 0.2))
            x, y = self._env.project_to_free(x, y)
            out[name] = (x, y)
        return out

    def _build_internal_env(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env = IsaacEnvironment(
            bounds_xy=self._bounds,
            dt_sec=self._dt,
            use_isaac_sim=False,
            require_isaac=False,
            rng_seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        self._env.load_world(str(self._world_cfg["world"]["scene_name"]))
        if self._fallback_obs:
            self._env.set_static_obstacles(self._fallback_obs)
        if self._ped_cfg:
            self._env.configure_pedestrians(self._ped_cfg)
        sp = self._spawn_dict()
        self._env.reset_episode(sp)

        self._robot_states = {}
        for name, (x, y) in sp.items():
            pos = np.array([x, y], dtype=float)
            self._robot_states[name] = RobotState(
                name=name,
                position=pos.copy(),
                target=pos.copy(),
                last_position=pos.copy(),
            )

    def _to_cell(self, point: np.ndarray) -> Tuple[int, int]:
        x_min, _, y_min, _ = self._bounds
        return (
            int((float(point[0]) - x_min) / self._grid_res),
            int((float(point[1]) - y_min) / self._grid_res),
        )

    def _obs_vector(self) -> np.ndarray:
        assert self._metrics is not None and self._env is not None
        vis = np.zeros((self._nx, self._ny), dtype=np.float32)
        for cx, cy in self._metrics.visited_cells:
            if 0 <= cx < self._nx and 0 <= cy < self._ny:
                vis[cx, cy] = 1.0
        flat = vis.reshape(-1)
        x_min, x_max, y_min, y_max = self._bounds
        rob = []
        for i in range(self._n_robots):
            name = f"robot_{i}"
            p = self._robot_states[name].position
            rob.extend(
                [
                    2.0 * (float(p[0]) - x_min) / (x_max - x_min + 1e-9) - 1.0,
                    2.0 * (float(p[1]) - y_min) / (y_max - y_min + 1e-9) - 1.0,
                ]
            )
        ped_feat: List[float] = []
        for ped in self._env._pedestrians[:8]:
            px, py = float(ped.position[0]), float(ped.position[1])
            ped_feat.extend(
                [
                    2.0 * (px - x_min) / (x_max - x_min + 1e-9) - 1.0,
                    2.0 * (py - y_min) / (y_max - y_min + 1e-9) - 1.0,
                ]
            )
        while len(ped_feat) < 16:
            ped_feat.append(0.0)
        step_n = np.array(
            [2.0 * self._step_count / max(self._max_episode_steps, 1) - 1.0],
            dtype=np.float32,
        )
        out = np.concatenate([flat, np.array(rob, dtype=np.float32), np.array(ped_feat[:16], dtype=np.float32), step_n])
        return out.astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_cov = 0.0
        self._build_internal_env(seed)
        self._metrics = CoverageMetrics(bounds_xy=self._bounds, grid_resolution_m=self._grid_res)
        blocked = set()
        x_min, _, y_min, _ = self._bounds
        assert self._env is not None
        for ox, oy, _ in self._env.get_obstacle_disks():
            cx = int((ox - x_min) / self._grid_res)
            cy = int((oy - y_min) / self._grid_res)
            blocked.add((cx, cy))
        self._metrics.set_blocked_cells(blocked)
        for state in self._robot_states.values():
            self._metrics.visited_cells.add(self._to_cell(state.position))
        self._prev_cov = self._metrics.coverage_percent()
        return self._obs_vector(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._env is not None and self._metrics is not None
        self._step_count += 1
        for i in range(self._n_robots):
            name = f"robot_{i}"
            idx = int(action[i])
            dx, dy = _action_to_delta(idx)
            pos = self._robot_states[name].position
            tx = float(pos[0] + dx * self._step_scale)
            ty = float(pos[1] + dy * self._step_scale)
            target = np.array([tx, ty], dtype=float)
            self._env.set_robot_target(name, target)
            self._robot_states[name].target = target.copy()

        self._env.step_simulation()

        for name in self._robot_states:
            self._robot_states[name].update_position(self._env.get_robot_pose(name))

        visited_before = len(self._metrics.visited_cells)
        self._metrics.update(self._robot_states)
        visited_after = len(self._metrics.visited_cells)
        new_cells = max(0, visited_after - visited_before)

        cov = self._metrics.coverage_percent()
        delta = cov - self._prev_cov
        self._prev_cov = cov

        reward = delta * self._r_cov_scale + new_cells * self._r_new_cell - self._r_step
        min_clear = self._env.min_robot_pedestrian_clearance_m()
        collision = False
        if min_clear < 0.0:
            reward -= self._r_collision
            collision = True
        elif min_clear < self._near_thresh:
            reward -= self._r_near

        terminated = cov >= self._target_cov
        truncated = self._step_count >= self._max_episode_steps
        if terminated:
            reward += self._r_success

        info = {
            "coverage": cov,
            "min_pedestrian_clearance_m": min_clear,
            "collision": collision,
            "n_steps": self._step_count,
        }
        return self._obs_vector(), float(reward), terminated, truncated, info


def load_rl_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("rl_training") or {}
