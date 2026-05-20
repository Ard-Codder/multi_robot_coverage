from __future__ import annotations

import json
from pathlib import Path

import pytest

from coverage_lab.isaac_export.replay_data import (
    LabReplayData,
    assert_paths_align_for_replay,
    build_lab_replay_data,
    iter_replay_frame_indices,
    parse_obstacles_from_result,
    path_length_stats,
)


def _minimal_result() -> dict:
    return {
        "scene": "test_scene",
        "bounds_xy": [-5.0, 5.0, -4.0, 4.0],
        "dt_sec": 0.1,
        "obstacles": [
            {"type": "disk", "x": 0.0, "y": 0.0, "r": 1.0},
            {"type": "rect", "x": 2.0, "y": 1.0, "w": 1.0, "h": 2.0},
        ],
        "robot_paths": {
            "robot_0": [[0.0, 0.1], [0.1, 0.1], [0.2, 0.1]],
            "robot_1": [[-0.1, 0.0], [-0.1, 0.1], [-0.1, 0.2]],
        },
        "pedestrian_paths": {"ped_0": [[1.0, 1.0], [1.0, 1.1], [1.0, 1.2]]},
    }


def test_parse_obstacles_disk_and_rect() -> None:
    disks, rects = parse_obstacles_from_result(_minimal_result())
    assert len(disks) == 1 and disks[0].r == 1.0
    assert len(rects) == 1 and rects[0].w == 1.0 and rects[0].h == 2.0


def test_build_lab_replay_data_from_json_only() -> None:
    data = build_lab_replay_data(_minimal_result())
    assert isinstance(data, LabReplayData)
    assert data.bounds_xy == (-5.0, 5.0, -4.0, 4.0)
    assert data.scene_name == "test_scene"
    assert len(data.disks) == 1 and len(data.rectangles) == 1
    assert set(data.robot_paths) == {"robot_0", "robot_1"}


def test_build_lab_replay_data_with_geometry_yaml() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "experiments_lab" / "scenes" / "large_complex_dynamic.yaml"
    if not yaml_path.is_file():
        pytest.skip("scene yaml not present")
    base = _minimal_result()
    data = build_lab_replay_data(base, geometry_yaml=yaml_path)
    assert data.scene_name == "large_complex_dynamic"
    assert len(data.rectangles) >= 1


def test_iter_replay_frame_indices_stride_max() -> None:
    assert iter_replay_frame_indices(10, stride=3, max_steps=None) == [0, 3, 6, 9]
    assert iter_replay_frame_indices(10, stride=1, max_steps=4) == [0, 1, 2, 3]


def test_path_length_stats() -> None:
    m, lens = path_length_stats(
        {"robot_0": [[0, 0]] * 5},
        {"ped_0": [[1, 1]] * 5},
    )
    assert m == 5
    assert lens["robot:robot_0"] == 5


def test_assert_paths_align_for_replay() -> None:
    data = build_lab_replay_data(_minimal_result())
    assert assert_paths_align_for_replay(data, stride=2, max_steps=None) == 3
    assert len(iter_replay_frame_indices(3, stride=2)) == 2


def test_assert_paths_mismatch_raises() -> None:
    bad = _minimal_result()
    bad["pedestrian_paths"] = {"ped_0": [[0, 0]]}
    data = build_lab_replay_data(bad)
    with pytest.raises(ValueError, match="pedestrian"):
        assert_paths_align_for_replay(data)


def test_fixture_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "run.json"
    p.write_text(json.dumps(_minimal_result()), encoding="utf-8")
    data = build_lab_replay_data(json.loads(p.read_text(encoding="utf-8")))
    assert len(data.robot_paths["robot_0"]) == 3
