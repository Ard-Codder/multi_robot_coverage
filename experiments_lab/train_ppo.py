from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("experiments_lab/scenes/dynamic_B_long.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--out", type=Path, default=Path("results/lab/rl/ppo_dynamic_B_long"))
    ap.add_argument("--mode", choices=["smoke", "train"], default="train")
    ap.add_argument("--init-model", type=Path, default=None, help="Warm-start from existing PPO model zip")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from coverage_lab.io import load_scene_yaml
    from coverage_lab.rl.env_gym import CoveragePedEnv

    scene = load_scene_yaml((ROOT / args.scene).resolve())
    if not scene.pedestrians:
        raise SystemExit("PPO training expects a dynamic scene with pedestrians.")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env(seed_off: int = 0):
        env = CoveragePedEnv(scene=scene, seed=int(args.seed) + seed_off)
        return Monitor(env)

    env = DummyVecEnv([lambda: make_env(0)])
    eval_env = DummyVecEnv([lambda: make_env(10_000)])
    log_dir = out_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    n_steps = 256 if args.mode == "smoke" else 1024
    batch_size = 128 if args.mode == "smoke" else 512
    total_steps = min(int(args.timesteps), 20_000) if args.mode == "smoke" else int(args.timesteps)

    if args.init_model is not None:
        init_path = (ROOT / args.init_model).resolve() if not args.init_model.is_absolute() else args.init_model
        model = PPO.load(str(init_path), env=env, device="cpu")
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=2.5e-4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            tensorboard_log=str(log_dir),
            device="cpu",
        )

    ckpt_cb = CheckpointCallback(save_freq=20_000, save_path=str(out_dir / "checkpoints"), name_prefix="ppo_ckpt")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=10_000,
        deterministic=True,
        n_eval_episodes=5,
    )
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, eval_cb], progress_bar=False)

    model.save(str(out_dir / "ppo_model.zip"))
    best_zip = out_dir / "best" / "best_model.zip"
    if best_zip.exists():
        print(f"Best model: {best_zip}")
    print(f"Saved: {out_dir / 'ppo_model.zip'}")


if __name__ == "__main__":
    main()

