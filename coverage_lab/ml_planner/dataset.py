from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Sample:
    obs: np.ndarray  # (H,W) visited mask
    action: int  # 0..4


def _discretize_move(p0: np.ndarray, p1: np.ndarray) -> int:
    d = p1 - p0
    if float(np.linalg.norm(d)) < 1e-6:
        return 4
    if abs(float(d[0])) > abs(float(d[1])):
        return 2 if float(d[0]) > 0 else 3
    return 0 if float(d[1]) > 0 else 1


def build_dataset_from_runs(
    json_paths: List[Path],
    *,
    window: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Supervised dataset: local visited map -> next move direction (from classical trajectories)."""
    X: List[np.ndarray] = []
    y: List[int] = []
    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        bounds = data["bounds_xy"]
        x_min, x_max, y_min, y_max = [float(v) for v in bounds]
        res = float(data["grid_resolution_m"])
        visited_cells = set(tuple(c) for c in data.get("visited_cells", []))
        robot_paths: Dict[str, List[List[float]]] = data.get("robot_paths", {})
        for pts in robot_paths.values():
            if not pts or len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=float)
            for i in range(len(arr) - 1):
                p0 = arr[i]
                p1 = arr[i + 1]
                cx = int((p0[0] - x_min) / res)
                cy = int((p0[1] - y_min) / res)
                w = int(window)
                r = w // 2
                obs = np.zeros((w, w), dtype=np.float32)
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        c = (cx + dx, cy + dy)
                        if c in visited_cells:
                            obs[dy + r, dx + r] = 1.0
                X.append(obs[None, :, :])  # add channel dim
                y.append(_discretize_move(p0, p1))
    if not X:
        return np.zeros((0, 1, window, window), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)

