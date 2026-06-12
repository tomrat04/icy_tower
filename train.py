"""Trening PPO (Stable-Baselines3) — curriculum 2 etapy, VecNormalize."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from icy_tower.callbacks import (
    IcyTowerEvalCallback,
    TensorboardGameCallback,
    VecNormalizeSaveCallback,
)
from icy_tower.env import IcyTowerEnv
from icy_tower.observations import OBS_DIM
from icy_tower.wrappers import IcyTowerVecWrapper

CURRICULUM_PHASES = [
    {"name": "etap1", "start_min": 0, "start_max": 100, "jump_per_step": True},
    {"name": "etap2", "start_min": 75, "start_max": 175, "jump_per_step": False},
]

CHECKPOINT_STEPS = 50_000
EVAL_STEPS = 50_000


def _make_env(seed: int, phase: dict) -> IcyTowerVecWrapper:
    return IcyTowerVecWrapper(
        Monitor(
            IcyTowerEnv(
                seed=seed,
                start_level_min=phase["start_min"],
                start_level_max=phase["start_max"],
                enable_jump_per_step=phase["jump_per_step"],
            ),
            info_keywords=("highest_level", "start_level"),
        )
    )


def _apply_curriculum(vec_env, phase: dict) -> None:
    vec_env.env_method(
        "set_curriculum",
        phase["start_min"],
        phase["start_max"],
        phase["jump_per_step"],
    )


def train(
    timesteps: int = 3_333_333,
    additional_timesteps: int | None = None,
    n_envs: int = 8,
    save_dir: str = "models",
    tb_log: str = "logs/tensorboard",
    run_name: str | None = None,
    load_model: str | None = None,
    finetune_lr: float | None = None,
    seed: int = 42,
    n_eval_episodes: int = 20,
    vec_normalize_path: str | None = None,
) -> None:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    run = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_path = Path(tb_log)
    phase_steps = timesteps // len(CURRICULUM_PHASES)
    checkpoint_freq = max(CHECKPOINT_STEPS // n_envs, 1)
    eval_freq = max(EVAL_STEPS // n_envs, 1)

    def _make_phase0():
        return _make_env(seed, CURRICULUM_PHASES[0])

    train_env = make_vec_env(_make_phase0, n_envs=n_envs, seed=seed)
    vec_path = Path(vec_normalize_path) if vec_normalize_path else save_dir_path / "vecnormalize.pkl"

    if load_model and vec_path.exists():
        train_env = VecNormalize.load(str(vec_path), train_env)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_base = DummyVecEnv([lambda: _make_env(seed + 1, CURRICULUM_PHASES[-1])])
    eval_env = VecNormalize(
        eval_base,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    print(f"Obserwacja: {OBS_DIM} cech")
    print("Algorytm: PPO | curriculum 2 etapy | VecNormalize obs+reward")

    if load_model:
        model_path = Path(load_model)
        if not model_path.suffix:
            model_path = Path(str(model_path) + ".zip")
        model = PPO.load(model_path, env=train_env)
        print(f"Wczytano model: {model_path}")
        print(f"  num_timesteps w pliku: {model.num_timesteps:,}")
        if finetune_lr is not None:
            model.learning_rate = finetune_lr
            print(f"Learning rate: {finetune_lr}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.4,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tb_log_path),
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        )

    vec_save_cb = VecNormalizeSaveCallback(
        train_env, save_dir_path, save_freq=checkpoint_freq
    )
    checkpoint = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_dir_path),
        name_prefix="icy_ppo",
    )
    eval_cb = IcyTowerEvalCallback(
        eval_env,
        train_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(save_dir_path / "best"),
        vec_save_cb=vec_save_cb,
    )
    game_cb = TensorboardGameCallback()
    callbacks = CallbackList([checkpoint, vec_save_cb, eval_cb, game_cb])

    if load_model and additional_timesteps is not None:
        total_timesteps = model.num_timesteps + additional_timesteps
        print(f"Dotrenowanie: {model.num_timesteps:,} → {total_timesteps:,} (+{additional_timesteps:,})")
        phase_plan = [(CURRICULUM_PHASES[-1], additional_timesteps)]
    else:
        total_timesteps = timesteps
        if load_model:
            print(f"Dotrenowanie: {model.num_timesteps:,} → {total_timesteps:,}")
        phase_plan = [(p, phase_steps) for p in CURRICULUM_PHASES]
        remainder = timesteps - phase_steps * len(CURRICULUM_PHASES)
        if remainder > 0:
            phase_plan[-1] = (phase_plan[-1][0], phase_plan[-1][1] + remainder)

    print("=" * 50)
    print(f"Cel treningu:     {total_timesteps:,} kroków środowiska")
    print(f"Równoległe envy:  {n_envs}")
    print(f"Eval co ~:        {eval_freq * n_envs:,} kroków środowiska")
    print(f"Checkpoint co ~:  {CHECKPOINT_STEPS:,} kroków środowiska")
    for phase, steps in phase_plan:
        print(
            f"  {phase['name']}: start {phase['start_min']}–{phase['start_max']}, "
            f"jump/step={'tak' if phase['jump_per_step'] else 'nie'}, "
            f"{steps:,} kroków"
        )
    print("=" * 50)

    reset_timesteps = load_model is None
    for phase, steps in phase_plan:
        _apply_curriculum(train_env, phase)
        print(f"\n>>> {phase['name']}: start {phase['start_min']}–{phase['start_max']}")
        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            tb_log_name=run,
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )
        reset_timesteps = False

    model.save(save_dir_path / "icy_ppo_final")
    vec_save_cb.save_now()
    train_env.close()
    eval_env.close()

    print(f"Model zapisany w: {save_dir_path / 'icy_ppo_final'}")
    print(f"VecNormalize:     {save_dir_path / 'vecnormalize.pkl'}")
    print(f"TensorBoard:      tensorboard --logdir {tb_log_path.resolve()}")
    print(f"Run:              {run}")


if __name__ == "__main__":
    train(
        load_model="models/best/best_model",      # albo models/icy_ppo_final
        additional_timesteps=1_000_000,           # ile kroków DALEJ
        vec_normalize_path="models/vecnormalize.pkl",
        finetune_lr=5e-5,                         # opcjonalnie, niższy LR
        save_dir="models",
        tb_log="logs/tensorboard",
        run_name="etap2_resume",
        n_envs=8,                                 # jak przy pierwszym treningu
    )