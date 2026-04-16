import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coverage_sim.metrics.coverage_metrics import CoverageMetrics
from coverage_sim.robots.robot_state import RobotState
from coverage_sim.simulation.simulation_loop import run_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_load_balance_stats() -> None:
    m = CoverageMetrics(bounds_xy=(0.0, 10.0, 0.0, 10.0), grid_resolution_m=1.0)
    a = RobotState(
        name="robot_0",
        position=np.zeros(2),
        target=np.zeros(2),
        travelled_distance=10.0,
        last_position=np.zeros(2),
    )
    b = RobotState(
        name="robot_1",
        position=np.zeros(2),
        target=np.zeros(2),
        travelled_distance=30.0,
        last_position=np.zeros(2),
    )
    s = m.load_balance_stats({"robot_0": a, "robot_1": b})
    assert s["distance_mean_m"] == pytest.approx(20.0)
    assert s["load_balance_cv"] > 0.0


def test_run_experiment_fallback_short() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        out_json = base / "out.json"
        out_plot = base / "plot"
        r = run_experiment(
            "random_walk",
            out_json,
            out_plot,
            ROOT,
            experiment_seed=0,
            max_steps=80,
            target_coverage=0.99,
            use_isaac_sim=False,
            save_plots=False,
        )
        assert out_json.is_file()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert r["steps"] <= 80
    assert "load_balance_cv" in r
    assert "per_robot_distance_m" in r
    assert r.get("experiment_seed") == 0
    assert loaded["algorithm"] == "random_walk"
