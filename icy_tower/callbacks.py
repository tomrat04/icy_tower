from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from icy_tower.config import HARD_MODE_LEVEL, TRAIN_START_LEVEL_MAX


def stratified_eval_starts(n_episodes: int) -> list[int]:
    """Stały rozkład startów 0..175 co 10 pięter — mniej szumu między punktami eval."""
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


class IcyTowerEvalCallback(BaseCallback):
    """
    Eval ze stratified startami (nie losowa próbka) + EMA na wykresie.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        n_eval_episodes: int = 90,
        best_model_save_path: Optional[str] = None,
        eval_ema_alpha: float = 0.25,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.eval_ema_alpha = eval_ema_alpha
        self.best_mean_reward = -np.inf
        self._ema_reward: Optional[float] = None
        self._ema_levels_up: Optional[float] = None
        self._eval_round = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        if self.model is None:
            return True

        start_levels = stratified_eval_starts(self.n_eval_episodes)
        rewards: list[float] = []
        lengths: list[int] = []
        levels_up: list[int] = []
        easy_rewards: list[float] = []
        hard_rewards: list[float] = []

        for ep_i, start_level in enumerate(start_levels):
            obs, info = self.eval_env.reset(
                options={
                    "start_level": start_level,
                    "episode_seed": 50_000 + self._eval_round * 10_000 + ep_i,
                }
            )
            start_level = int(info.get("start_level", start_level))
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1

            peak = int(info.get("peak_level", start_level))
            landed = int(info.get("highest_level", start_level))
            progress = max(peak, landed) - start_level

            rewards.append(ep_reward)
            lengths.append(ep_len)
            levels_up.append(progress)
            if start_level < HARD_MODE_LEVEL:
                easy_rewards.append(ep_reward)
            else:
                hard_rewards.append(ep_reward)

        self._eval_round += 1

        mean_reward = float(np.mean(rewards))
        mean_length = float(np.mean(lengths))
        mean_levels_up = float(np.mean(levels_up))

        a = self.eval_ema_alpha
        self._ema_reward = (
            mean_reward
            if self._ema_reward is None
            else a * mean_reward + (1.0 - a) * self._ema_reward
        )
        self._ema_levels_up = (
            mean_levels_up
            if self._ema_levels_up is None
            else a * mean_levels_up + (1.0 - a) * self._ema_levels_up
        )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/mean_levels_up", mean_levels_up)
        self.logger.record("eval/mean_reward_ema", self._ema_reward)
        self.logger.record("eval/mean_levels_up_ema", self._ema_levels_up)

        if easy_rewards:
            self.logger.record("eval/mean_reward_easy", float(np.mean(easy_rewards)))
        if hard_rewards:
            self.logger.record("eval/mean_reward_hard", float(np.mean(hard_rewards)))

        if self.verbose > 0:
            easy_n, hard_n = len(easy_rewards), len(hard_rewards)
            print(
                f"Eval @ {self.num_timesteps}: reward={mean_reward:.1f} "
                f"(ema {self._ema_reward:.1f}), "
                f"levels_up={mean_levels_up:.1f} (ema {self._ema_levels_up:.1f}), "
                f"len={mean_length:.0f} (easy n={easy_n}, hard n={hard_n})"
            )

        if mean_reward > self.best_mean_reward and self.best_model_save_path is not None:
            self.best_mean_reward = mean_reward
            path = self.best_model_save_path / "best_model"
            self.model.save(path)
            if self.verbose > 0:
                print(f"  Nowy best model → {path}.zip")

        return True
