from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ACTION_LR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
_ACTION_UD = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
# row i = remap for action class i (faster than dict + tolist on huge y)
_LUT_LR = [_ACTION_LR[i] for i in range(5)]
_LUT_UD = [_ACTION_UD[i] for i in range(5)]


def _print_torch_env() -> None:
    import sys

    import torch

    ver = torch.__version__
    cuda_ok = torch.cuda.is_available()
    cuda_build = getattr(torch.version, "cuda", None)
    print(f"[env] python: {sys.executable}", flush=True)
    print(f"[env] torch {ver}  cuda_available={cuda_ok}  torch.version.cuda={cuda_build}", flush=True)
    if cuda_ok:
        print(f"[env] gpu: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        if "+cpu" in ver:
            print(
                "[env] Похоже, стоит CPU-колёса PyTorch. Для NVIDIA (в т.ч. RTX 50xx) установи CUDA-сборку в ЭТОТ же интерпретатор:\n"
                '  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128',
                flush=True,
            )
        else:
            print(
                "[env] CUDA не видна PyTorch (другой Python / драйвер / переменные). Проверь:\n"
                '  where python\n'
                '  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
                flush=True,
            )


def _collect_json_paths(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        root = (ROOT / d).resolve() if not d.is_absolute() else d.resolve()
        if not root.is_dir():
            print(f"[warn] skip missing runs dir: {root}")
            continue
        out.extend(sorted(root.glob("*.json")))
    return out


def _apply_flip_augment(X: "torch.Tensor", y: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Mirror flips + action relabeling (local patch symmetry)."""
    import torch

    lut_lr = torch.tensor(_LUT_LR, dtype=y.dtype, device=y.device)
    lut_ud = torch.tensor(_LUT_UD, dtype=y.dtype, device=y.device)
    xs = [X]
    ys = [y]
    o = X[:, 0, :, :]
    xs.append(torch.flip(o, dims=[2]).unsqueeze(1))
    ys.append(lut_lr[y])
    xs.append(torch.flip(o, dims=[1]).unsqueeze(1))
    ys.append(lut_ud[y])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train SmallCNN for MLGuidedPlanner from experiment JSON (robot_paths + visited_cells).",
    )
    ap.add_argument(
        "--runs-dir",
        action="append",
        dest="runs_dirs",
        type=Path,
        help="Directory with *.json lab runs (repeat for multiple dirs). Default: results/lab/classic_static",
    )
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--input-mode", choices=("legacy", "rich"), default="legacy")
    ap.add_argument("--model-variant", choices=("small", "spatial"), default="small")
    ap.add_argument("--epochs", type=int, default=12, help="Epochs when not using --max-time-minutes.")
    ap.add_argument("--max-time-minutes", type=float, default=0, help="If > 0, train until elapsed time (ignores --epochs).")
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=1_000_000,
        help="Safety cap when training by time.",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--aug-flips", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: CUDA if available. cuda: fail if CUDA missing.",
    )
    ap.add_argument("--out", type=Path, default=Path("results/lab/ml_guided/model.pt"))
    args = ap.parse_args()

    import torch

    _print_torch_env()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[env] Запрошен --device cuda, но torch.cuda.is_available() == False.")
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from coverage_lab.ml_planner.dataset import build_dataset_from_runs
    from coverage_lab.ml_planner.model import SmallCNN

    runs_dirs = args.runs_dirs if args.runs_dirs else [Path("results/lab/classic_static")]
    json_paths = _collect_json_paths(runs_dirs)
    if not json_paths:
        raise SystemExit(f"No json files under {runs_dirs}")

    print(f"Using {len(json_paths)} JSON runs from:")
    for d in runs_dirs:
        print(f"  - {d}")

    X_np, y_np = build_dataset_from_runs(
        [Path(p) for p in json_paths],
        window=int(args.window),
        input_mode=str(args.input_mode),
    )
    if len(X_np) == 0:
        raise SystemExit("Empty dataset; need runs with robot_paths/visited_cells.")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    print(f"device={device}  samples={len(X_np)}  (data on CPU RAM; batches -> {device})")

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).to(dtype=torch.long)
    del X_np, y_np
    if args.aug_flips:
        X, y = _apply_flip_augment(X, y)
        print(f"after flip aug: {len(X)} samples")

    ds = TensorDataset(X, y)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=pin,
        num_workers=0,
    )
    model = SmallCNN(in_ch=int(X.shape[1]), num_actions=5, variant=str(args.model_variant)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12)
    loss_fn = nn.CrossEntropyLoss()

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_time = float(args.max_time_minutes) > 0
    use_epochs = not use_time
    target_epochs = int(args.epochs) if use_epochs else int(args.max_epochs)

    t0 = time.monotonic()
    best_loss = float("inf")
    best_state = None
    epoch = 0

    def one_epoch() -> float:
        model.train()
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
        return total / max(n, 1)

    while True:
        epoch += 1
        avg = one_epoch()
        sched.step(avg)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "window": int(args.window),
                    "in_ch": int(X.shape[1]),
                    "input_mode": str(args.input_mode),
                    "model_variant": str(args.model_variant),
                },
                out_path,
            )
            print(f"  (checkpoint -> {out_path.name} best_loss={best_loss:.5f})", flush=True)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch} loss={avg:.5f} lr={lr:.2e} elapsed_min={(time.monotonic() - t0) / 60:.1f}")

        if use_epochs and epoch >= target_epochs:
            break
        if use_time:
            if epoch >= int(args.max_epochs):
                break
            if (time.monotonic() - t0) >= float(args.max_time_minutes) * 60.0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "window": int(args.window),
            "in_ch": int(X.shape[1]),
            "input_mode": str(args.input_mode),
            "model_variant": str(args.model_variant),
        },
        out_path,
    )
    print(f"Saved: {out_path}  (best_loss={best_loss:.5f})")


if __name__ == "__main__":
    main()
