import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_sim.simulation.simulation_loop import run_experiment


if __name__ == "__main__":
    result = run_experiment(
        algorithm_name="frontier",
        output_json=ROOT / "results" / "frontier.json",
        output_plot_prefix=ROOT / "results" / "frontier",
        root_dir=ROOT,
    )
    print(f"Frontier completed: coverage={result['coverage_percent']:.3f}")

