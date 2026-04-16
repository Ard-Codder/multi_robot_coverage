from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np

BoundsXY = Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max


@dataclass
class DiskObstacle:
    x: float
    y: float
    r: float


@dataclass
class RectObstacle:
    """Axis-aligned rectangle obstacle (center + width/height)."""

    x: float
    y: float
    w: float
    h: float


@dataclass
class PedSpec:
    id: str
    x: float
    y: float
    goal_b_x: float
    goal_b_y: float
    radius_m: float = 0.35
    max_speed_mps: float = 1.1


@dataclass
class LabScene:
    """Serializable scene config for CoverageLab."""

    name: str
    bounds_xy: BoundsXY
    dt_sec: float
    grid_resolution_m: float
    num_robots: int
    max_steps: int
    target_coverage: float
    obstacles: List[DiskObstacle]
    rectangles: List[RectObstacle]
    pedestrians: List[PedSpec]


AlgorithmName = Literal[
    "baseline_random_walk",
    "baseline_grid",
    "baseline_voronoi",
    "baseline_frontier",
    "boustrophedon",
    "stc",
    "darp_boustro",
    "darp_stc",
    "rl_ppo",
    "ml_guided",
]


RobotPaths = Dict[str, List[List[float]]]  # {robot_i: [[x,y],...]}
PedPaths = Dict[str, List[List[float]]]


@dataclass
class StepState:
    robot_xy: Dict[str, np.ndarray]  # name -> [x,y]
    ped_xy: Dict[str, np.ndarray]  # id -> [x,y]

