"""Trening PPO (Stable-Baselines3) — losowy start 0→175, wygrana 200."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from icy_tower.callbacks import IcyTowerEvalCallback, TensorboardGameCallback
from icy_tower.env import IcyTowerEnv
from icy_tower.observations import OBS_DIM


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening PPO — Icy Tower")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Docelowa liczba kroków (od zera) lub — przy --load-model bez --additional-timesteps — kontynuacja DO tej wartości",
    )
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        default=None,
        help="Przy --load-model: trenuj tyle kroków DALEJ (np. 500000 → z 2M do 2.5M)",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--tb-log", type=str, default="logs/tensorboard")
    parser.add_argument("--run-name", type=str, default=None, help="Nazwa runu w TensorBoard")
    parser.add_argument("--load-model", type=str, default=None, help="Kontynuuj z zapisanego modelu (.zip)")
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Niższy learning rate przy dotrenowaniu (np. 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=50,
        help="Epizody na jeden punkt eval (więcej = mniej szumu na wykresie)",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = Path(args.tb_log)

    def _make():
        return Monitor(
            IcyTowerEnv(seed=args.seed),
            info_keywords=("highest_level", "start_level"),
        )

    train_env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(
        IcyTowerEnv(seed=args.seed + 1),
        info_keywords=("highest_level", "start_level"),
    )

    print(f"Obserwacja: {OBS_DIM} cech")
    print("Algorytm: PPO | losowy start 0→175, wygrana 200 (~43% epizodów na hard)")

    if args.load_model:
        model_path = Path(args.load_model)
        if not model_path.suffix:
            model_path = Path(str(model_path) + ".zip")
        model = PPO.load(model_path, env=train_env)
        print(f"Wczytano model: {model_path}")
        print(f"  num_timesteps w pliku: {model.num_timesteps:,}")
        if "best" in model_path.parts:
            print(
                "  (best_model = najlepszy wynik eval, nie koniec treningu — "
                "użyj models/icy_ppo_final.zip dla ostatniego kroku)"
            )
        if args.finetune_lr is not None:
            model.learning_rate = args.finetune_lr
            print(f"Learning rate: {args.finetune_lr}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(tb_log),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        )

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(save_dir),
        name_prefix="icy_ppo",
    )
    eval_cb = IcyTowerEvalCallback(
        eval_env,
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=str(save_dir / "best"),
    )
    game_cb = TensorboardGameCallback()

    if args.load_model and args.additional_timesteps is not None:
        total_timesteps = model.num_timesteps + args.additional_timesteps
        print(f"Dotrenowanie: {model.num_timesteps:,} → {total_timesteps:,} (+{args.additional_timesteps:,})")
    else:
        total_timesteps = args.timesteps
        if args.load_model:
            print(f"Dotrenowanie: {model.num_timesteps:,} → {total_timesteps:,}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint, eval_cb, game_cb],
        tb_log_name=run_name,
        reset_num_timesteps=args.load_model is None,
        progress_bar=True,
    )
    model.save(save_dir / "icy_ppo_final")
    train_env.close()
    eval_env.close()

    print(f"Model zapisany w: {save_dir / 'icy_ppo_final'}")
    print(f"TensorBoard:  tensorboard --logdir {tb_log.resolve()}")
    print(f"Run:          {run_name}")


if __name__ == "__main__":
    main()
