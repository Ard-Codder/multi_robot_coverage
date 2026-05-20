from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_json_paths(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        root = (ROOT / d).resolve() if not d.is_absolute() else d.resolve()
        if not root.exists():
            print(f"[warn] missing runs dir: {root}")
            continue
        out.extend(sorted(root.glob("*.json")))
    return out


def _print_torch_env() -> None:
    import torch

    print(f"[env] python: {sys.executable}", flush=True)
    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} build={torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] gpu: {torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ML goal/frontier policy.")
    ap.add_argument("--runs-dir", action="append", dest="runs_dirs", type=Path)
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--label-mode", choices=("teacher_frontier", "best_of_k"), default="teacher_frontier")
    ap.add_argument("--rollout-k", type=int, default=16)
    ap.add_argument("--max-samples-per-run", type=int, default=20_000)
    ap.add_argument("--max-samples", type=int, default=300_000)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-time-minutes", type=float, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--out", type=Path, default=Path("results/lab/ml_goal/model.pt"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from coverage_lab.ml_planner.goal_dataset import build_goal_dataset_from_runs
    from coverage_lab.ml_planner.model import GoalCNN

    _print_torch_env()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is unavailable.")
    device = torch.device("cpu" if args.device == "cpu" else "cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dirs = args.runs_dirs or [Path("results/lab/ml_teacher_large_complex")]
    json_paths = _collect_json_paths(dirs)
    if not json_paths:
        raise SystemExit("No JSON runs found.")
    print(f"Using {len(json_paths)} JSON runs")

    X_np, y_np = build_goal_dataset_from_runs(
        [Path(p) for p in json_paths],
        window=int(args.window),
        stride=int(args.stride),
        max_samples_per_run=int(args.max_samples_per_run),
        label_mode=str(args.label_mode),
        rollout_k=int(args.rollout_k),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if len(X_np) == 0:
        raise SystemExit("Empty goal dataset.")
    print(f"dataset={X_np.shape} labels={y_np.shape} device={device}")

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).to(dtype=torch.long)
    del X_np, y_np
    ds = TensorDataset(X, y)
    val_n = max(1, int(len(ds) * float(args.val_ratio)))
    train_n = max(1, len(ds) - val_n)
    if train_n + val_n > len(ds):
        val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(int(args.seed)))
    train_dl = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, pin_memory=pin, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, pin_memory=pin, num_workers=0)

    model = GoalCNN(in_ch=int(X.shape[1]), window=int(args.window)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state = None
    t0 = time.monotonic()
    epoch = 0

    def run_epoch() -> float:
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
        return total / max(n, 1)

    def validate() -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                loss = loss_fn(model(xb), yb)
                total += float(loss.item()) * int(xb.shape[0])
                n += int(xb.shape[0])
        return total / max(n, 1)

    while True:
        epoch += 1
        train_loss = run_epoch()
        val_loss = validate()
        elapsed = (time.monotonic() - t0) / 60.0
        print(f"epoch {epoch} train={train_loss:.5f} val={val_loss:.5f} elapsed_min={elapsed:.1f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "window": int(args.window),
                    "in_ch": int(X.shape[1]),
                    "model": "GoalCNN",
                    "target": "local_goal_heatmap",
                    "label_mode": str(args.label_mode),
                    "rollout_k": int(args.rollout_k),
                    "val_loss": float(best_val),
                },
                out_path,
            )
            print(f"  checkpoint -> {out_path} val={best_val:.5f}", flush=True)
        if float(args.max_time_minutes) > 0 and elapsed >= float(args.max_time_minutes):
            break
        if float(args.max_time_minutes) <= 0 and epoch >= int(args.epochs):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "window": int(args.window),
            "in_ch": int(X.shape[1]),
            "model": "GoalCNN",
            "target": "local_goal_heatmap",
            "label_mode": str(args.label_mode),
            "rollout_k": int(args.rollout_k),
            "val_loss": float(best_val),
        },
        out_path,
    )
    print(f"Saved: {out_path} best_val={best_val:.5f}")


if __name__ == "__main__":
    main()
