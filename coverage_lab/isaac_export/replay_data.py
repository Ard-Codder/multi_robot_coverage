from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from coverage_lab.io import load_scene_yaml
from coverage_lab.types import BoundsXY, DiskObstacle, LabScene, RectObstacle


@dataclass(frozen=True)
class LabReplayData:
    """Everything needed to replay a lab run in 3D (kinematic puppet)."""

    scene_name: str
    bounds_xy: BoundsXY
    dt_sec: float
    grid_resolution_m: Optional[float]
    disks: Tuple[DiskObstacle, ...]
    rectangles: Tuple[RectObstacle, ...]
    robot_paths: Dict[str, List[List[float]]]
    pedestrian_paths: Dict[str, List[List[float]]]


def _bounds_from_result(data: Dict[str, Any]) -> BoundsXY:
    b = data.get("bounds_xy")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if isinstance(b, dict):
        return (
            float(b["x_min"]),
            float(b["x_max"]),
            float(b["y_min"]),
            float(b["y_max"]),
        )
    raise ValueError("result JSON must contain bounds_xy as [x_min,x_max,y_min,y_max] or dict keys")


def parse_obstacles_from_result(data: Dict[str, Any]) -> Tuple[List[DiskObstacle], List[RectObstacle]]:
    """Parse `obstacles` list from coverage_lab result JSON (typed disk/rect or legacy disks)."""
    raw = data.get("obstacles") or []
    disks: List[DiskObstacle] = []
    rects: List[RectObstacle] = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        if "w" in o and "h" in o:
            rects.append(
                RectObstacle(
                    x=float(o["x"]),
                    y=float(o["y"]),
                    w=float(o["w"]),
                    h=float(o["h"]),
                )
            )
        elif "r" in o:
            disks.append(DiskObstacle(x=float(o["x"]), y=float(o["y"]), r=float(o["r"])))
    return disks, rects


def path_length_stats(
    robot_paths: Dict[str, List[List[float]]],
    pedestrian_paths: Dict[str, List[List[float]]],
) -> Tuple[int, Dict[str, int]]:
    """Return (min_length, per_key_lengths) over non-empty paths."""
    lengths: Dict[str, int] = {}
    for name, pts in robot_paths.items():
        lengths[f"robot:{name}"] = len(pts)
    for name, pts in pedestrian_paths.items():
        lengths[f"ped:{name}"] = len(pts)
    nonempty = [n for n, L in lengths.items() if L > 0]
    if not nonempty:
        return 0, lengths
    m = min(lengths[k] for k in nonempty)
    return m, lengths


def iter_replay_frame_indices(
    min_path_length: int,
    *,
    stride: int = 1,
    max_steps: Optional[int] = None,
) -> List[int]:
    """Frame indices into robot_paths[*][i] aligned with pedestrian_paths."""
    if min_path_length <= 0:
        return []
    L = min_path_length
    if max_steps is not None:
        L = min(L, int(max_steps))
    st = max(1, int(stride))
    return list(range(0, L, st))


def build_lab_replay_data(
    result: Dict[str, Any],
    *,
    geometry_yaml: Optional[Path] = None,
) -> LabReplayData:
    """
    Build replay pack from a `coverage_lab` result dict.

    If `geometry_yaml` is set, static bounds and obstacles come from that scene file;
    trajectories still come from `result` (caller must ensure consistency).
    """
    robot_paths = {str(k): list(v) for k, v in (result.get("robot_paths") or {}).items()}
    pedestrian_paths = {str(k): list(v) for k, v in (result.get("pedestrian_paths") or {}).items()}

    if geometry_yaml is not None:
        scene = load_scene_yaml(Path(geometry_yaml))
        bounds = scene.bounds_xy
        grid_resolution_m: Optional[float] = float(scene.grid_resolution_m)
        disks, rects = list(scene.obstacles), list(scene.rectangles)
        scene_name = scene.name
    else:
        bounds = _bounds_from_result(result)
        disks, rects = parse_obstacles_from_result(result)
        scene_name = str(result.get("scene", "unknown"))
        try:
            grid_resolution_m = float(result.get("grid_resolution_m"))
            if grid_resolution_m <= 0.0:
                grid_resolution_m = None
        except (TypeError, ValueError):
            grid_resolution_m = None
        disks = list(disks)
        rects = list(rects)

    dt = float(result.get("dt_sec", 0.1))
    return LabReplayData(
        scene_name=scene_name,
        bounds_xy=bounds,
        dt_sec=dt,
        grid_resolution_m=grid_resolution_m,
        disks=tuple(disks),
        rectangles=tuple(rects),
        robot_paths=robot_paths,
        pedestrian_paths=pedestrian_paths,
    )


def assert_paths_align_for_replay(
    data: LabReplayData,
    *,
    stride: int = 1,
    max_steps: Optional[int] = None,
) -> int:
    """Return robot-path length used for replay; raises if paths are unusable."""
    r_lens = [len(p) for p in data.robot_paths.values()]
    if not r_lens or min(r_lens) < 1:
        raise ValueError("robot_paths must be non-empty for replay")
    m = min(r_lens)
    if max(r_lens) != m:
        raise ValueError(f"robot path lengths differ: { {k: len(v) for k, v in data.robot_paths.items()} }")
    for name, p in data.pedestrian_paths.items():
        if len(p) not in (0, m):
            raise ValueError(f"pedestrian {name} length {len(p)} != robot length {m}")
    idx = iter_replay_frame_indices(m, stride=stride, max_steps=max_steps)
    if not idx:
        raise ValueError("no replay frames after stride/max_steps")
    return m
