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

CURRICULUM_PHASES = [
    {"name": "etap1", "start_min": 0, "start_max": 100, "jump_per_step": True},
    {"name": "etap2", "start_min": 75, "start_max": 175, "jump_per_step": False},
]

CHECKPOINT_STEPS = 50000
EVAL_STEPS = 50000


def make_env(seed, phase):
    env = IcyTowerEnv(
        seed=seed,
        start_level_min=phase["start_min"],
        start_level_max=phase["start_max"],
        enable_jump_per_step=phase["jump_per_step"],
    )
    return IcyTowerVecWrapper(Monitor(env, info_keywords=("highest_level", "start_level")))


def apply_curriculum(vec_env, phase):
    vec_env.env_method("set_curriculum", phase["start_min"], phase["start_max"], phase["jump_per_step"])


def train(timesteps=3333333, n_envs=8, save_dir="models", tb_log="logs/tensorboard",
    run_name="trening", n_eval_episodes=20):

    os.makedirs(save_dir, exist_ok=True)
    run = run_name

    phase_steps = timesteps // len(CURRICULUM_PHASES)
    checkpoint_freq = max(CHECKPOINT_STEPS // n_envs, 1)
    eval_freq = max(EVAL_STEPS // n_envs, 1)

    train_env = make_vec_env(lambda: make_env(CURRICULUM_PHASES[0]), n_envs=n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_base = DummyVecEnv([lambda: make_env(CURRICULUM_PHASES[-1])])
    eval_env = VecNormalize(eval_base, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy",train_env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=512,
        n_epochs=6,
        gamma=0.9999,
        clip_range=0.1,
        ent_coef=0.02,
        vf_coef=0.5,
        tensorboard_log=tb_log,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    vec_save_cb = VecNormalizeSaveCallback(train_env, save_dir, save_freq=checkpoint_freq)
    checkpoint = CheckpointCallback(save_freq=checkpoint_freq, save_path=save_dir, name_prefix="icy_ppo")
    eval_cb = IcyTowerEvalCallback(
        eval_env, train_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        best_model_save_path=f"{save_dir}/best", vec_save_cb=vec_save_cb,
    )
    callbacks = CallbackList([checkpoint, vec_save_cb, eval_cb, TensorboardGameCallback()])

    phase_plan = [(p, phase_steps) for p in CURRICULUM_PHASES]
    reszta = timesteps - phase_steps * len(CURRICULUM_PHASES)
    if reszta > 0:
        phase_plan[-1] = (phase_plan[-1][0], phase_plan[-1][1] + reszta)

    for phase, steps in phase_plan:
        apply_curriculum(train_env, phase)
        print(f"\ntrening: {phase['name']} - {steps} kroków")
        model.learn(total_timesteps=steps, callback=callbacks, tb_log_name=run, progress_bar=True)

    model.save(f"{save_dir}/icy_ppo_final")
    vec_save_cb.save_now()
    train_env.close()
    eval_env.close()
    print(f"model: {save_dir}/icy_ppo_final.zip")

train()