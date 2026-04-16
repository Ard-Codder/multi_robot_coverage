from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/lab/classic_static"))
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/lab/ml_guided/model.pt"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from coverage_lab.ml_planner.dataset import build_dataset_from_runs
    from coverage_lab.ml_planner.model import SmallCNN

    runs_dir = (ROOT / args.runs_dir).resolve()
    json_paths = sorted(runs_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No json files in {runs_dir}")

    X, y = build_dataset_from_runs([Path(p) for p in json_paths], window=int(args.window))
    if len(X) == 0:
        raise SystemExit("Empty dataset; need classic runs with robot_paths/visited_cells.")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    model = SmallCNN(in_ch=1, num_actions=5)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(int(args.epochs)):
        total = 0.0
        n = 0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
        print(f"epoch {ep+1}/{args.epochs} loss={total/max(n,1):.4f}")

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "window": int(args.window)}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

