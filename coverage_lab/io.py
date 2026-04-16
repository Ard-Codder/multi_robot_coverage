from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from coverage_lab.types import DiskObstacle, LabScene, PedSpec, RectObstacle


def load_scene_yaml(path: Path) -> LabScene:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sc = data.get("scene") or {}
    b = sc.get("bounds_xy") or {}
    obstacles = [
        DiskObstacle(x=float(o["x"]), y=float(o["y"]), r=float(o["r"]))
        for o in (sc.get("obstacles") or [])
    ]
    rectangles = [
        RectObstacle(
            x=float(r["x"]),
            y=float(r["y"]),
            w=float(r["w"]),
            h=float(r["h"]),
        )
        for r in (sc.get("rectangles") or [])
    ]
    peds = [
        PedSpec(
            id=str(p["id"]),
            x=float(p["x"]),
            y=float(p["y"]),
            goal_b_x=float(p["goal_b_x"]),
            goal_b_y=float(p["goal_b_y"]),
            radius_m=float(p.get("radius_m", 0.35)),
            max_speed_mps=float(p.get("max_speed_mps", 1.1)),
        )
        for p in (sc.get("pedestrians") or [])
    ]
    return LabScene(
        name=str(sc.get("name", path.stem)),
        bounds_xy=(float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"])),
        dt_sec=float(sc.get("dt_sec", 0.1)),
        grid_resolution_m=float(sc.get("grid_resolution_m", 0.5)),
        num_robots=int(sc.get("num_robots", 3)),
        max_steps=int(sc.get("max_steps", 1200)),
        target_coverage=float(sc.get("target_coverage", 0.95)),
        obstacles=obstacles,
        rectangles=rectangles,
        pedestrians=peds,
    )


def save_result_json(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def load_result_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

