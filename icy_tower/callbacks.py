from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from icy_tower.config import HARD_MODE_LEVEL


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
    Eval z rozbiciem na easy (start < 100) i hard (start >= 100).
    Mniej mylące niż samo eval/mean_reward przy losowym starcie 0–175.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        n_eval_episodes: int = 50,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        if self.model is None:
            return True

        rewards: list[float] = []
        lengths: list[int] = []
        levels_up: list[int] = []
        easy_rewards: list[float] = []
        hard_rewards: list[float] = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            start_level = int(info.get("start_level", 0))
            ep_reward = 0.0
            ep_len = 0
            peak_level = start_level
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                peak_level = max(peak_level, int(info.get("highest_level", peak_level)))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            levels_up.append(peak_level - start_level)
            if start_level < HARD_MODE_LEVEL:
                easy_rewards.append(ep_reward)
            else:
                hard_rewards.append(ep_reward)

        mean_reward = float(np.mean(rewards))
        mean_length = float(np.mean(lengths))
        mean_levels_up = float(np.mean(levels_up))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/mean_levels_up", mean_levels_up)

        if easy_rewards:
            self.logger.record("eval/mean_reward_easy", float(np.mean(easy_rewards)))
        if hard_rewards:
            self.logger.record("eval/mean_reward_hard", float(np.mean(hard_rewards)))

        if self.verbose > 0:
            easy_n, hard_n = len(easy_rewards), len(hard_rewards)
            print(
                f"Eval @ {self.num_timesteps}: reward={mean_reward:.1f}, "
                f"len={mean_length:.0f}, levels_up={mean_levels_up:.1f} "
                f"(easy n={easy_n}, hard n={hard_n})"
            )

        if mean_reward > self.best_mean_reward and self.best_model_save_path is not None:
            self.best_mean_reward = mean_reward
            path = self.best_model_save_path / "best_model"
            self.model.save(path)
            if self.verbose > 0:
                print(f"  Nowy best model → {path}.zip")

        return True
