from __future__ import annotations

"""
1) Run classical teachers on large_complex_dynamic → JSON in output_dir from YAML.
2) Train SmallCNN on combined dirs (static + prior large_complex + new teacher runs) for --train-minutes.

Example:
  python experiments_lab/ml_teacher_pipeline.py
  python experiments_lab/ml_teacher_pipeline.py --skip-batch --train-minutes 90
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch-config",
        type=Path,
        default=Path("experiments_lab/batch_ml_teacher_large_complex.yaml"),
    )
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--train-minutes", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=Path("results/lab/ml_guided/model.pt"))
    args = ap.parse_args()

    cfg_path = (ROOT / args.batch_config).resolve()
    if not args.skip_batch:
        print("Starting teacher data batch...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-u",
                str(ROOT / "experiments_lab" / "run_batch.py"),
                "--config",
                str(cfg_path),
            ],
            cwd=str(ROOT),
        )

    # Mix small static corpus + previous final_large_complex + fresh teacher export
    teacher_dir = None
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        teacher_dir = Path(str(data.get("output_dir", "results/lab/ml_teacher_large_complex")))
    except (OSError, KeyError, ImportError):
        teacher_dir = Path("results/lab/ml_teacher_large_complex")

    train_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "experiments_lab" / "train_ml_guided.py"),
        "--runs-dir",
        "results/lab/classic_static",
        "--runs-dir",
        "results/lab/presentation_static",
        "--runs-dir",
        "results/lab/final_large_complex",
        "--runs-dir",
        str(teacher_dir),
        "--max-time-minutes",
        str(float(args.train_minutes)),
        "--out",
        str(args.out),
    ]
    print("Starting training (long run)...", flush=True)
    subprocess.check_call(train_cmd, cwd=str(ROOT))
    print("OK: model at", (ROOT / args.out).resolve())


if __name__ == "__main__":
    main()
