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


def _make_env(seed, phase):
    env = IcyTowerEnv(
        seed=seed,
        start_level_min=phase["start_min"],
        start_level_max=phase["start_max"],
        enable_jump_per_step=phase["jump_per_step"],
    )
    return IcyTowerVecWrapper(Monitor(env, info_keywords=("highest_level", "start_level")))


def _apply_curriculum(vec_env, phase):
    vec_env.env_method(
        "set_curriculum",
        phase["start_min"],
        phase["start_max"],
        phase["jump_per_step"],
    )


def train(
        timesteps=3333333,
        additional_timesteps=None,
        n_envs=8,
        save_dir="models",
        tb_log="logs/tensorboard",
        run_name=None,
        load_model=None,
        finetune_lr=None,
        seed=42,
        n_eval_episodes=20,
        vec_normalize_path=None,
):
    os.makedirs(save_dir, exist_ok=True)
    run = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    phase_steps = timesteps // len(CURRICULUM_PHASES)
    checkpoint_freq = max(CHECKPOINT_STEPS // n_envs, 1)
    eval_freq = max(EVAL_STEPS // n_envs, 1)

    train_env = make_vec_env(lambda: _make_env(seed, CURRICULUM_PHASES[0]), n_envs=n_envs, seed=seed)

    vec_path = vec_normalize_path if vec_normalize_path else f"{save_dir}/vecnormalize.pkl"

    if load_model and os.path.exists(vec_path):
        train_env = VecNormalize.load(vec_path, train_env)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_base = DummyVecEnv([lambda: _make_env(seed + 1, CURRICULUM_PHASES[-1])])
    eval_env = VecNormalize(eval_base, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    if load_model:
        model = PPO.load(load_model, env=train_env)
        print(f"Wczytano model z {load_model}. Kroków w pliku: {model.num_timesteps}")
        if finetune_lr:
            model.learning_rate = finetune_lr
            print(f"Nowy learning rate: {finetune_lr}")
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
            tensorboard_log=tb_log,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        )

    vec_save_cb = VecNormalizeSaveCallback(train_env, save_dir, save_freq=checkpoint_freq)
    checkpoint = CheckpointCallback(save_freq=checkpoint_freq, save_path=save_dir, name_prefix="icy_ppo")
    eval_cb = IcyTowerEvalCallback(
        eval_env, train_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        best_model_save_path=f"{save_dir}/best", vec_save_cb=vec_save_cb,
    )
    game_cb = TensorboardGameCallback()
    callbacks = CallbackList([checkpoint, vec_save_cb, eval_cb, game_cb])

    if load_model and additional_timesteps:
        print(f"Dotrenowanie o {additional_timesteps} kroków.")
        phase_plan = [(CURRICULUM_PHASES[-1], additional_timesteps)]
    else:
        phase_plan = [(p, phase_steps) for p in CURRICULUM_PHASES]
        reszta = timesteps - phase_steps * len(CURRICULUM_PHASES)
        if reszta > 0:
            phase_plan[-1] = (phase_plan[-1][0], phase_plan[-1][1] + reszta)

    reset_timesteps = load_model is None
    for phase, steps in phase_plan:
        _apply_curriculum(train_env, phase)
        print(f"\n--- Trening: {phase['name']} ({steps} kroków) ---")

        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            tb_log_name=run,
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )
        reset_timesteps = False

    model.save(f"{save_dir}/icy_ppo_final")
    vec_save_cb.save_now()
    train_env.close()
    eval_env.close()
    print(f"sciezka modelu: {save_dir}/icy_ppo_final.zip")

train(
    load_model="models/best/best_model",
    additional_timesteps=1000000,
    vec_normalize_path="models/vecnormalize.pkl",
    finetune_lr=5e-5,
    save_dir="models",
    tb_log="logs/tensorboard",
    run_name="etap2_resume",
    n_envs=8,
)