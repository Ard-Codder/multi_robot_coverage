"""
Обучение PPO для среды CoveragePedRLEnv (покрытие + умные пешеходы, kinematic).

Зависимости:
  pip install -r requirements-rl.txt

Пример:
  python experiments/train_rl_coverage_ppo.py --timesteps 300000
  python experiments/train_rl_coverage_ppo.py --timesteps 8000 --device cuda --no-vecnorm

Как читать лог TensorBoard:
  explained_variance высокий — критик хорошо предсказывает возврат, это не означает хорошую политику.
  Смотрите eval/mean_reward в TensorBoard (EvalCallback), лучшая модель в results/rl_models/best_model/.

Улучшения по умолчанию: VecNormalize(obs), EvalCallback, более глубокая MLP, n_steps=1024/env,
linear LR decay, сохранение VecNormalize рядом с весами.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# SB3 предупреждает, что MlpPolicy чаще гоняют на CPU — для GPU всё равно ок.
warnings.filterwarnings(
    "ignore",
    message=".*run PPO on the GPU.*MlpPolicy.*",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.utils import get_linear_fn
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as e:
        raise SystemExit(
            "Нужны stable-baselines3 и torch: pip install -r requirements-rl.txt"
        ) from e

    from coverage_sim.rl.coverage_ped_env import CoveragePedRLEnv

    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Шаги между оценками (по умолчанию max(8192, 4*n_steps*n_envs))",
    )
    p.add_argument(
        "--no-eval-callback",
        action="store_true",
        help="Не вызывать EvalCallback (быстрее для отладки)",
    )
    p.add_argument(
        "--no-vecnorm",
        action="store_true",
        help="Отключить нормализацию наблюдений (хуже сходимость на больших Box)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (CUDA если есть), cpu, cuda или cuda:0",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Параллельных сред (по умолчанию: 4 при GPU, 1 при CPU)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="По умолчанию: 64 на CPU, 256 при CUDA+несколько сред",
    )
    p.add_argument(
        "--vec",
        type=str,
        choices=("auto", "subproc", "dummy"),
        default="auto",
        help="auto: subprocess при n_envs>1",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=ROOT / "results" / "rl_models" / "ppo_coverage_ped.zip",
    )
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Устройство обучения: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA запрошена, но torch.cuda.is_available() == False.")

    n_envs = args.n_envs
    if n_envs is None:
        n_envs = 4 if device.startswith("cuda") else 1
    n_envs = max(1, int(n_envs))

    n_steps = 1024
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 256 if (device.startswith("cuda") and n_envs >= 2) else 64

    def make_env():
        return CoveragePedRLEnv()

    if n_envs == 1:
        train_env = DummyVecEnv([make_env])
    else:
        use_subproc = args.vec == "subproc" or (args.vec == "auto" and n_envs > 1)
        if use_subproc:
            try:
                train_env = SubprocVecEnv([make_env] * n_envs)
            except Exception as e:
                print(f"SubprocVecEnv недоступен ({e}), DummyVecEnv × {n_envs}")
                train_env = DummyVecEnv([make_env] * n_envs)
        else:
            train_env = DummyVecEnv([make_env] * n_envs)

    if not args.no_vecnorm:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
        )

    eval_env = DummyVecEnv([lambda: CoveragePedRLEnv()])
    if not args.no_vecnorm:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
        )

    eval_freq = args.eval_freq
    if eval_freq is None:
        eval_freq = max(8192, 4 * n_steps * n_envs)

    callbacks = []
    if not args.no_eval_callback:
        best_dir = args.save.parent / "best_model"
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(args.save.parent / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=int(args.eval_episodes),
                deterministic=True,
                render=False,
            )
        )
        print(f"EvalCallback: каждые {eval_freq} шагов, {args.eval_episodes} эпизодов, лучшая модель -> {best_dir}")

    lr_schedule = get_linear_fn(3e-4, 1e-5, 1.0)

    print(
        f"Векторная среда: {n_envs} env, n_steps={n_steps}, batch_size={batch_size}, "
        f"VecNormalize={'да' if not args.no_vecnorm else 'нет'}"
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=device,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(ROOT / "results" / "rl_tensorboard"),
    )
    model.learn(
        total_timesteps=int(args.timesteps),
        progress_bar=False,
        callback=callbacks if callbacks else None,
    )

    save_path = args.save
    if save_path.suffix.lower() == ".zip":
        save_path = save_path.with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    if not args.no_vecnorm:
        train_env.save(str(save_path) + "_vecnormalize.pkl")
    print(f"Модель сохранена: {save_path}.zip")
    if not args.no_vecnorm:
        print(f"VecNormalize: {save_path}_vecnormalize.pkl")

    # Финальная оценка (загружаем лучшую, если есть)
    best_zip = args.save.parent / "best_model" / "best_model.zip"
    load_path = best_zip if best_zip.is_file() else Path(str(save_path) + ".zip")
    print(f"Финальный eval загружает: {load_path}")

    eval_single = CoveragePedRLEnv()
    # PPO.load ожидает путь к .zip (не обрезать суффикс).
    loaded = PPO.load(str(load_path), device=device)
    if not args.no_vecnorm:
        vn_path = str(save_path) + "_vecnormalize.pkl"
        if Path(vn_path).is_file():
            eval_wrapped = DummyVecEnv([lambda: CoveragePedRLEnv()])
            eval_wrapped = VecNormalize.load(vn_path, eval_wrapped)
            eval_wrapped.training = False
            eval_wrapped.norm_reward = False
            use_vec = True
        else:
            eval_wrapped = None
            use_vec = False
    else:
        eval_wrapped = None
        use_vec = False

    rewards = []
    covs = []
    for ep in range(int(args.eval_episodes)):
        if use_vec and eval_wrapped is not None:
            obs = eval_wrapped.reset()
            dones = np.zeros((1,), dtype=bool)
            total_r = 0.0
            steps = 0
            info: dict = {}
            while not dones[0]:
                action, _ = loaded.predict(obs, deterministic=True)
                obs, r, dones, infos = eval_wrapped.step(action)
                total_r += float(r[0])
                steps += 1
                if dones[0] and infos:
                    info = infos[0] if isinstance(infos, list) else infos
        else:
            obs, _ = eval_single.reset(seed=10_000 + ep)
            done = False
            total_r = 0.0
            steps = 0
            while not done:
                action, _ = loaded.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_single.step(action)
                total_r += float(r)
                steps += 1
                done = term or trunc
        rewards.append(total_r)
        covs.append(float(info.get("coverage", 0.0)))
        print(
            f"eval ep {ep+1}: return={total_r:.2f}, steps={steps}, coverage={covs[-1]:.3f}, "
            f"min_clear={info.get('min_pedestrian_clearance_m', 'n/a')}"
        )
    print(f"Средний return: {np.mean(rewards):.2f}, среднее coverage: {np.mean(covs):.3f}")


if __name__ == "__main__":
    main()
