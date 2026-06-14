import os
from datetime import datetime

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
from icy_tower.wrappers import IcyTowerVecWrapper

TARGET_PHASE = {"name": "etap2_finetune", "start_min": 75, "start_max": 175, "jump_per_step": False}

CHECKPOINT_STEPS = 50000
EVAL_STEPS = 50000


def _make_env(seed, phase):
    env = IcyTowerEnv(
        seed=seed,
        start_level_min=phase["start_min"],
        start_level_max=phase["start_max"],
        enable_jump_per_step=phase["jump_per_step"],
    )
    return IcyTowerVecWrapper(Monitor(env, info_keywords=("highest_level", "start_level")))


def _apply_curriculum(vec_env, phase):
    vec_env.env_method("set_curriculum", phase["start_min"], phase["start_max"], phase["jump_per_step"])


def finetune(
        model_path="models/best/best_model",
        vec_path="models/vecnormalize.pkl",
        timesteps=1000000,
        finetune_lr=5e-5,
        n_envs=8,
        save_dir="models",
        tb_log="logs/tensorboard",
        run_name=None,
        seed=42,
        n_eval_episodes=20,
):
    os.makedirs(save_dir, exist_ok=True)
    run = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_freq = max(CHECKPOINT_STEPS // n_envs, 1)
    eval_freq = max(EVAL_STEPS // n_envs, 1)

    train_env = make_vec_env(lambda: _make_env(seed, TARGET_PHASE), n_envs=n_envs, seed=seed)
    train_env = VecNormalize.load(vec_path, train_env)

    eval_base = DummyVecEnv([lambda: _make_env(seed + 1, TARGET_PHASE)])
    eval_env = VecNormalize(eval_base, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = PPO.load(model_path, env=train_env)
    print(f"wczytano model {model_path}, wznowienie od kroku: {model.num_timesteps}")
    if finetune_lr:
        model.learning_rate = finetune_lr

    vec_save_cb = VecNormalizeSaveCallback(train_env, save_dir, save_freq=checkpoint_freq)
    checkpoint = CheckpointCallback(save_freq=checkpoint_freq, save_path=save_dir, name_prefix="icy_ppo")
    eval_cb = IcyTowerEvalCallback(
        eval_env, train_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        best_model_save_path=f"{save_dir}/best", vec_save_cb=vec_save_cb,
    )
    callbacks = CallbackList([checkpoint, vec_save_cb, eval_cb, TensorboardGameCallback()])

    _apply_curriculum(train_env, TARGET_PHASE)
    print(f"\ntrening finetune: {timesteps} kroków")

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=run,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    model.save(f"{save_dir}/icy_ppo_finetuned")
    vec_save_cb.save_now()
    train_env.close()
    eval_env.close()
    print(f"Gotowe! Model: {save_dir}/icy_ppo_finetuned.zip")


finetune(
    model_path="models/best/best_model",
    vec_path="models/vecnormalize.pkl",
    timesteps=1000000,
    finetune_lr=5e-5,
    run_name="etap2_resume",
)