from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from coverage_lab.algorithms.baselines import BaselineFrontier
from coverage_lab.env.grid_world import GridWorld2D, WorldState
from coverage_lab.io import load_scene_yaml
from coverage_lab.sim import run_experiment_lab
from coverage_lab.types import DiskObstacle

ROOT = Path(__file__).resolve().parents[1]


def test_large_complex_scene_loads() -> None:
    scene = load_scene_yaml(ROOT / "experiments_lab" / "scenes" / "large_complex_dynamic.yaml")

    assert scene.name == "large_complex_dynamic"
    assert scene.num_robots == 8
    assert len(scene.rectangles) >= 8
    assert len(scene.pedestrians) >= 4


def test_final_large_complex_smoke_config() -> None:
    data = yaml.safe_load((ROOT / "experiments_lab" / "batch_final_large_complex_smoke.yaml").read_text(encoding="utf-8"))

    assert data["scene_path"] == "experiments_lab/scenes/large_complex_dynamic.yaml"
    assert "baseline_frontier" in data["matrix"]["algorithms"]


def test_ml_teacher_large_complex_batch_config() -> None:
    data = yaml.safe_load((ROOT / "experiments_lab" / "batch_ml_teacher_large_complex.yaml").read_text(encoding="utf-8"))

    assert data["scene_path"] == "experiments_lab/scenes/large_complex_dynamic.yaml"
    sr = data["matrix"]["seeds_range"]
    assert isinstance(sr, (list, tuple)) and len(sr) == 2
    algos = data["matrix"]["algorithms"]
    assert "ml_guided" not in [a.lower() for a in algos]


def test_ml_ablation_large_complex_batch_config() -> None:
    data = yaml.safe_load((ROOT / "experiments_lab" / "batch_ml_ablation_large_complex.yaml").read_text(encoding="utf-8"))

    algos = [a.lower() for a in data["matrix"]["algorithms"]]
    assert "baseline_voronoi" in algos
    assert "ml_guided_pure" in algos
    assert "ml_guided_soft_guarded" in algos
    assert "ml_guided_guarded" in algos


def test_ml_goal_ablation_large_complex_batch_config() -> None:
    data = yaml.safe_load((ROOT / "experiments_lab" / "batch_ml_goal_ablation_large_complex.yaml").read_text(encoding="utf-8"))

    algos = [a.lower() for a in data["matrix"]["algorithms"]]
    assert "baseline_voronoi" in algos
    assert "ml_goal_pure" in algos
    assert "ml_goal_soft_guarded" in algos
    assert "ml_goal_guarded" in algos


def test_large_complex_short_simulation() -> None:
    scene = load_scene_yaml(ROOT / "experiments_lab" / "scenes" / "large_complex_dynamic.yaml")
    scene.max_steps = 40
    scene.target_coverage = 0.99

    result = run_experiment_lab(
        algorithm=BaselineFrontier(seed=0),
        algorithm_name="baseline_frontier",
        scene=scene,
        seed=0,
        keep_paths=False,
    )

    assert result["scene"] == "large_complex_dynamic"
    assert result["steps"] <= 40
    assert result["coverage_percent"] >= 0.0
    assert "robot_ped_violations" in result


def test_grid_world_side_step_around_disk_obstacle() -> None:
    world = GridWorld2D(
        bounds_xy=(-2.0, 2.0, -2.0, 2.0),
        dt_sec=0.1,
        grid_resolution_m=0.25,
        obstacles=[DiskObstacle(x=0.0, y=0.0, r=0.4)],
        rectangles=[],
        pedestrians=[],
        seed=0,
        max_step_m=0.25,
        robot_radius_m=0.1,
    )
    start = np.array([-0.65, 0.0], dtype=float)
    state = WorldState(
        robot_xy={"robot_0": start.copy()},
        robot_target_xy={"robot_0": np.array([1.0, 0.0], dtype=float)},
        pedestrians=[],
    )

    new_state = world.step(state, {"robot_0": np.array([1.0, 0.0], dtype=float)})

    assert new_state.blocked_moves == 0
    assert float(np.linalg.norm(new_state.robot_xy["robot_0"] - start)) > 0.0
    assert abs(float(new_state.robot_xy["robot_0"][1])) > 0.0


def test_rich_ml_dataset_channels(tmp_path: Path) -> None:
    from coverage_lab.ml_planner.dataset import build_dataset_from_runs

    sample = {
        "bounds_xy": [0, 3, 0, 3],
        "grid_resolution_m": 1.0,
        "obstacle_cells": [[1, 1]],
        "robot_paths": {"robot_0": [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]},
    }
    path = tmp_path / "run.json"
    import json

    path.write_text(json.dumps(sample), encoding="utf-8")
    X, y = build_dataset_from_runs([path], window=3, input_mode="rich")

    assert X.shape == (2, 4, 3, 3)
    assert y.tolist() == [2, 2]
    assert X[:, 1].sum() >= 1.0  # blocked channel
    assert X[:, 3].sum() == 2.0  # robot-center channel


def test_goal_dataset_and_model_smoke(tmp_path: Path) -> None:
    import json

    import torch

    from coverage_lab.ml_planner.goal_dataset import build_goal_dataset_from_runs
    from coverage_lab.ml_planner.model import GoalCNN

    sample = {
        "bounds_xy": [0, 5, 0, 5],
        "grid_resolution_m": 1.0,
        "obstacle_cells": [[2, 2]],
        "robot_paths": {"robot_0": [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]]},
    }
    path = tmp_path / "goal_run.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    X, y = build_goal_dataset_from_runs([path], window=5, stride=1)
    assert X.shape[1:] == (6, 5, 5)
    assert len(y) == len(X)

    model = GoalCNN(in_ch=6, window=5)
    logits = model(torch.from_numpy(X[:2]))
    assert logits.shape == (min(2, len(X)), 25)


def test_final_demo_report_builder(tmp_path: Path) -> None:
    from scripts.build_final_demo_report import build_report

    summary = tmp_path / "summary.csv"
    fields = [
        "algorithm",
        "seed",
        "scene",
        "coverage_percent",
        "time_to_coverage_sec",
        "distance_travelled_m",
        "efficiency",
        "steps",
        "load_balance_cv",
        "blocked_moves",
        "min_pedestrian_clearance_m",
        "robot_robot_collisions",
        "robot_ped_violations",
        "json_path",
    ]
    rows = [
        {
            "algorithm": "ml_guided",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.78,
            "time_to_coverage_sec": 80.0,
            "distance_travelled_m": 1300.0,
            "efficiency": 0.0006,
            "steps": 800,
            "load_balance_cv": 0.3,
            "blocked_moves": 1000,
            "min_pedestrian_clearance_m": 0.1,
            "robot_robot_collisions": 0,
            "robot_ped_violations": 20,
            "json_path": "dummy.json",
        },
        {
            "algorithm": "baseline_frontier",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.78,
            "time_to_coverage_sec": 120.0,
            "distance_travelled_m": 1500.0,
            "efficiency": 0.0005,
            "steps": 1200,
            "load_balance_cv": 0.4,
            "blocked_moves": 2000,
            "min_pedestrian_clearance_m": 0.08,
            "robot_robot_collisions": 0,
            "robot_ped_violations": 40,
            "json_path": "dummy.json",
        },
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "report"
    build_report(summary, out_dir, sync_docs=False)

    assert (out_dir / "aggregate_metrics.csv").exists()
    report = (out_dir / "FINAL_SYSTEM_DEMO.md").read_text(encoding="utf-8")
    assert "Финальный демонстрационный пакет" in report
    assert "`ml_guided`" in report
    assert "VLA" in report


def test_ml_guided_diagnosis_builder(tmp_path: Path) -> None:
    from scripts.build_ml_guided_diagnosis import build_diagnosis

    summary = tmp_path / "summary.csv"
    fields = [
        "algorithm",
        "seed",
        "scene",
        "coverage_percent",
        "time_to_coverage_sec",
        "distance_travelled_m",
        "blocked_moves",
        "robot_ped_violations",
        "ml_model_step_rate",
        "ml_fallback_rate",
        "ml_unsafe_model_targets",
    ]
    rows = [
        {
            "algorithm": "baseline_voronoi",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.78,
            "time_to_coverage_sec": 80.0,
            "distance_travelled_m": 1300.0,
            "blocked_moves": 1000,
            "robot_ped_violations": 20,
            "ml_model_step_rate": "",
            "ml_fallback_rate": "",
            "ml_unsafe_model_targets": "",
        },
        {
            "algorithm": "ml_guided_pure",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.01,
            "time_to_coverage_sec": "",
            "distance_travelled_m": 0.0,
            "blocked_moves": 0,
            "robot_ped_violations": 100,
            "ml_model_step_rate": 1.0,
            "ml_fallback_rate": 0.0,
            "ml_unsafe_model_targets": 0,
        },
        {
            "algorithm": "ml_guided_soft_guarded",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.25,
            "time_to_coverage_sec": "",
            "distance_travelled_m": 9000.0,
            "blocked_moves": 800,
            "robot_ped_violations": 80,
            "ml_model_step_rate": 1.0,
            "ml_fallback_rate": 0.2,
            "ml_unsafe_model_targets": 10,
        },
        {
            "algorithm": "ml_guided_guarded",
            "seed": 0,
            "scene": "large_complex_dynamic",
            "coverage_percent": 0.78,
            "time_to_coverage_sec": 80.0,
            "distance_travelled_m": 1300.0,
            "blocked_moves": 1000,
            "robot_ped_violations": 20,
            "ml_model_step_rate": 0.0,
            "ml_fallback_rate": 1.0,
            "ml_unsafe_model_targets": 0,
        },
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out = build_diagnosis(summary, tmp_path / "diag", sync_docs=False)
    text = out.read_text(encoding="utf-8")
    assert "текущая обученная модель" in text
    assert "guardrail" in text


def test_ml_goal_report_builder(tmp_path: Path) -> None:
    from scripts.build_ml_goal_report import build_report

    summary = tmp_path / "summary.csv"
    fields = [
        "algorithm",
        "coverage_percent",
        "time_to_coverage_sec",
        "distance_travelled_m",
        "blocked_moves",
        "robot_ped_violations",
        "ml_goal_model_rate",
        "ml_goal_fallback_rate",
        "ml_goal_unsafe_goals",
        "ml_goal_avg_l1_distance",
    ]
    rows = [
        {
            "algorithm": "baseline_voronoi",
            "coverage_percent": 0.78,
            "time_to_coverage_sec": 80,
            "distance_travelled_m": 1300,
            "blocked_moves": 100,
            "robot_ped_violations": 10,
            "ml_goal_model_rate": "",
            "ml_goal_fallback_rate": "",
            "ml_goal_unsafe_goals": "",
            "ml_goal_avg_l1_distance": "",
        },
        {
            "algorithm": "ml_goal_soft_guarded",
            "coverage_percent": 0.76,
            "time_to_coverage_sec": 100,
            "distance_travelled_m": 1400,
            "blocked_moves": 120,
            "robot_ped_violations": 12,
            "ml_goal_model_rate": 1.0,
            "ml_goal_fallback_rate": 0.2,
            "ml_goal_unsafe_goals": 3,
            "ml_goal_avg_l1_distance": 8.5,
        },
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out = build_report(summary, tmp_path / "goal_report", sync_docs=False)
    text = out.read_text(encoding="utf-8")
    assert "ML-goal ablation" in text
    assert "ml_goal_soft_guarded" in text
