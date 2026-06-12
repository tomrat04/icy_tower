from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from icy_tower.config import HARD_MODE_LEVEL, TRAIN_START_LEVEL_MAX


def stratified_eval_starts(n_episodes: int) -> list[int]:
    """Stały rozkład startów 0..175 co 10 pięter."""
    levels = list(range(0, TRAIN_START_LEVEL_MAX + 1, 10))
    if levels[-1] != TRAIN_START_LEVEL_MAX:
        levels.append(TRAIN_START_LEVEL_MAX)
    return [levels[i % len(levels)] for i in range(n_episodes)]


class TensorboardGameCallback(BaseCallback):
    """Loguje highest_level z epizodu treningowego do TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if ep is None:
                continue
            level = ep.get("highest_level")
            if level is not None:
                self.logger.record("game/highest_level", float(level))
        return True


class VecNormalizeSaveCallback(BaseCallback):
    """Zapisuje statystyki VecNormalize przy checkpointach."""

    def __init__(self, vec_env: VecNormalize, save_path: Path, save_freq: int):
        super().__init__()
        self.vec_env = vec_env
        self.save_path = Path(save_path)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.save_now()
        return True

    def save_now(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.vec_env.save(str(self.save_path / "vecnormalize.pkl"))

    def save_best_copy(self, best_dir: Path) -> None:
        best_dir.mkdir(parents=True, exist_ok=True)
        self.vec_env.save(str(best_dir / "vecnormalize.pkl"))


class IcyTowerEvalCallback(BaseCallback):
    """Eval ze stratified startami. Best model po mean_levels_up; wzrost ent_coef po szczycie."""

    def __init__(
        self,
        eval_env: VecEnv,
        train_vec_env: VecNormalize,
        eval_freq: int,
        n_eval_episodes: int = 20,
        best_model_save_path: Optional[str] = None,
        ent_coef_bump: float = 0.002,
        ent_coef_max: float = 0.06,
        vec_save_cb: Optional[VecNormalizeSaveCallback] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_vec_env = train_vec_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.ent_coef_bump = ent_coef_bump
        self.ent_coef_max = ent_coef_max
        self.vec_save_cb = vec_save_cb
        self.best_mean_levels_up = -np.inf
        self._eval_round = 0

    def _sync_eval_normalization(self) -> None:
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.train_vec_env.obs_rms
            self.eval_env.ret_rms = self.train_vec_env.ret_rms

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        if self.model is None:
            return True

        self._sync_eval_normalization()

        start_levels = stratified_eval_starts(self.n_eval_episodes)
        rewards: list[float] = []
        lengths: list[int] = []
        levels_up: list[int] = []
        easy_rewards: list[float] = []
        hard_rewards: list[float] = []

        for ep_i, start_level in enumerate(start_levels):
            episode_seed = 50_000 + self._eval_round * 10_000 + ep_i
            self.eval_env.env_method(
                "set_eval_episode", start_level, episode_seed, indices=[0]
            )
            obs = self.eval_env.reset()
            ep_reward = 0.0
            ep_len = 0
            done = False
            last_info: dict = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                ep_reward += float(reward[0])
                ep_len += 1
                done = bool(dones[0])
                last_info = infos[0]

            ep_start = int(last_info.get("start_level", start_level))
            peak = int(last_info.get("peak_level", ep_start))
            landed = int(last_info.get("highest_level", ep_start))
            progress = max(peak, landed) - ep_start

            rewards.append(ep_reward)
            lengths.append(ep_len)
            levels_up.append(progress)
            if ep_start < HARD_MODE_LEVEL:
                easy_rewards.append(ep_reward)
            else:
                hard_rewards.append(ep_reward)

        self._eval_round += 1

        mean_reward = float(np.mean(rewards))
        mean_length = float(np.mean(lengths))
        mean_levels_up = float(np.mean(levels_up))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/mean_levels_up", mean_levels_up)
        self.logger.record("train/ent_coef", float(self.model.ent_coef))

        if easy_rewards:
            self.logger.record("eval/mean_reward_easy", float(np.mean(easy_rewards)))
        if hard_rewards:
            self.logger.record("eval/mean_reward_hard", float(np.mean(hard_rewards)))

        if self.verbose > 0:
            easy_n, hard_n = len(easy_rewards), len(hard_rewards)
            print(
                f"Eval @ {self.num_timesteps}: reward={mean_reward:.1f}, "
                f"levels_up={mean_levels_up:.1f}, "
                f"len={mean_length:.0f} (easy n={easy_n}, hard n={hard_n})"
            )

        if mean_levels_up > self.best_mean_levels_up and self.best_model_save_path is not None:
            self.best_mean_levels_up = mean_levels_up
            path = self.best_model_save_path / "best_model"
            self.model.save(path)
            if self.vec_save_cb is not None:
                self.vec_save_cb.save_best_copy(self.best_model_save_path)
            self.model.ent_coef = min(
                float(self.model.ent_coef) + self.ent_coef_bump, self.ent_coef_max
            )
            if self.verbose > 0:
                print(
                    f"  Nowy best model → {path}.zip (levels_up={mean_levels_up:.1f}, "
                    f"ent_coef={float(self.model.ent_coef):.4f})"
                )

        return True
