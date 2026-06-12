"""Opakowanie zewnętrzne — metody curriculum/eval widoczne przez Monitor i VecEnv."""

from __future__ import annotations

import gymnasium as gym

from icy_tower.env import IcyTowerEnv


class IcyTowerVecWrapper(gym.Wrapper):
    def _icy_env(self) -> IcyTowerEnv:
        env: gym.Env | None = self
        while env is not None:
            if isinstance(env, IcyTowerEnv):
                return env
            env = getattr(env, "env", None)
        raise RuntimeError("Nie znaleziono IcyTowerEnv w stosie wrapperów")

    def set_curriculum(
        self, start_level_min: int, start_level_max: int, enable_jump_per_step: bool
    ) -> None:
        self._icy_env().set_curriculum(
            start_level_min, start_level_max, enable_jump_per_step
        )

    def set_eval_episode(self, start_level: int, episode_seed: int) -> None:
        self._icy_env().set_eval_episode(start_level, episode_seed)
