import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_sim.simulation.simulation_loop import run_experiment


if __name__ == "__main__":
    print("Starting Isaac live run...")
    print("Expected in Stage: /World/Robots/robot_0..robot_2")
    result = run_experiment(
        algorithm_name="random_walk",
        output_json=ROOT / "results" / "random_walk_live.json",
        output_plot_prefix=ROOT / "results" / "random_walk_live",
        root_dir=ROOT,
        keep_visualization_open_sec=120.0,
        require_isaac=True,
    )
    print(
        "Live Random Walk completed: "
        f"coverage={result['coverage_percent']:.3f}, "
        f"distance={result['distance_travelled_m']:.1f}m"
    )

