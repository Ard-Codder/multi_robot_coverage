import json

from agent.tools import extract_and_run_tool_calls, run_coverage_experiment, run_isaac_live_script


def test_run_coverage_experiment_returns_json() -> None:
    s = run_coverage_experiment(
        algorithm="grid",
        seed=0,
        max_steps=50,
        target_coverage=0.99,
        use_isaac_sim=False,
        save_plots=False,
        output_name="test_agent_grid",
    )
    data = json.loads(s)
    assert "error" not in data
    assert data.get("algorithm") == "grid"
    assert "coverage_percent" in data


def test_run_isaac_live_rejects_unknown_script() -> None:
    s = run_isaac_live_script("experiments/not_allowed.py")
    data = json.loads(s)
    assert "error" in data


def test_extract_tool_calls() -> None:
    text = '<tool_call>run_coverage_experiment{"algorithm": "grid", "seed": 0, "max_steps": 20}</tool_call>'
    log = extract_and_run_tool_calls(text)
    assert "grid" in log
    assert "coverage_percent" in log or "error" in log
