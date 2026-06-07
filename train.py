"""Trening DQN (Stable-Baselines3) z logowaniem TensorBoard."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from icy_tower.callbacks import TensorboardGameCallback
from icy_tower.env import IcyTowerEnv
from icy_tower.observations import OBS_DIM


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening DQN — Icy Tower")
    parser.add_argument("--timesteps", type=int, default=3_333_333)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--tb-log", type=str, default="logs/tensorboard")
    parser.add_argument("--run-name", type=str, default=None, help="Nazwa runu w TensorBoard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = Path(args.tb_log)

    def _make():
        return Monitor(
            IcyTowerEnv(seed=args.seed),
            info_keywords=("highest_level", "score"),
        )

    train_env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(
        IcyTowerEnv(seed=args.seed + 1),
        info_keywords=("highest_level", "score"),
    )

    print(f"Obserwacja: {OBS_DIM} cech")

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=2.5e-4,
        buffer_size=400_000,
        learning_starts=20_000,
        batch_size=256,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.25,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(tb_log),
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(save_dir),
        name_prefix="icy_dqn",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval"),
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    game_cb = TensorboardGameCallback()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint, eval_cb, game_cb],
        tb_log_name=run_name,
        progress_bar=True,
    )
    model.save(save_dir / "icy_dqn_final")
    train_env.close()
    eval_env.close()

    print(f"Model zapisany w: {save_dir}")
    print(f"TensorBoard:  tensorboard --logdir {tb_log.resolve()}")
    print(f"Run:          {run_name}")


if __name__ == "__main__":
    main()
