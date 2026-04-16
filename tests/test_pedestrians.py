from coverage_sim.env.smart_pedestrians import parse_pedestrian_config, step_smart_pedestrian
from coverage_sim.env.smart_pedestrians import SmartPedestrian
import numpy as np


def test_parse_disabled() -> None:
    assert parse_pedestrian_config(None) == []
    assert parse_pedestrian_config({"enabled": False}) == []


def test_parse_two_agents() -> None:
    cfg = {
        "enabled": True,
        "defaults": {"radius_m": 0.4, "max_speed_mps": 1.0},
        "agents": [
            {"id": "a", "x": 0, "y": 0, "goal_b_x": 2, "goal_b_y": 0},
            {"id": "b", "x": 1, "y": 1, "goal_a_x": 0, "goal_a_y": 0, "goal_b_x": 3, "goal_b_y": 1},
        ],
    }
    peds = parse_pedestrian_config(cfg)
    assert len(peds) == 2
    assert peds[0].pid == "a"


def test_step_moves() -> None:
    ped = SmartPedestrian(
        pid="p",
        position=np.array([0.0, 0.0]),
        velocity=np.zeros(2),
        goal_a=np.array([0.0, 0.0]),
        goal_b=np.array([2.0, 0.0]),
        going_to_b=True,
        radius=0.3,
        max_speed=0.5,
    )
    beh = {"goal_switch_m": 0.1}
    step_smart_pedestrian(
        ped,
        dt=0.1,
        robot_positions=[],
        robot_radius=0.2,
        peer_positions=[],
        static_centers_radii=[],
        cfg=beh,
    )
    assert float(ped.position[0]) > 0.01
