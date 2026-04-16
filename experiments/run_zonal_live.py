import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_sim.simulation.simulation_loop import run_experiment


if __name__ == "__main__":
    print("Starting Isaac zonal live run...")
    result = run_experiment(
        algorithm_name="zonal",
        output_json=ROOT / "results" / "zonal_live.json",
        output_plot_prefix=ROOT / "results" / "zonal_live",
        root_dir=ROOT,
        keep_visualization_open_sec=120.0,
        require_isaac=True,
    )
    print(
        "Live Zonal completed: "
        f"coverage={result['coverage_percent']:.3f}, "
        f"distance={result['distance_travelled_m']:.1f}m"
    )

