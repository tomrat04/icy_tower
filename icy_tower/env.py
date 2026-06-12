from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from icy_tower.config import (
    DT,
    HARD_MODE_LEVEL,
    LEVEL_HEIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TRAIN_START_LEVEL_MAX,
    WIN_LEVEL,
)
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import OBS_DIM, build_observation

# --- Nagrody i kary ---
REWARD_LEVEL_UP = 10.0
REWARD_LEVEL_UP_HARD_BONUS = 5.0
REWARD_HIGHEST_LEVEL_BONUS = 0.2
REWARD_TAKEOFF = 0.2
REWARD_JUMP_PER_STEP = 0.05
MAX_JUMP_REWARD_PER_AIR = 0.6
PENALTY_IDLE_ON_START = 0.03
PENALTY_STAGNATION = 0.05
PENALTY_NEAR_BOTTOM = 0.015
PENALTY_PER_STEP = 0.001
PENALTY_DEATH = 25.0
REWARD_WIN = 500.0
START_LEVEL_IDLE_LIMIT = 45
STAGNATION_STEP_LIMIT = 60


class IcyTowerEnv(gym.Env):
    """
    Środowisko Gymnasium dla agenta RL.

    Akcje: 0 — brak, 1 — lewo, 2 — prawo, 3 — skok, 4 — lewo+skok, 5 — prawo+skok.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        max_episode_steps: int = 12_000,
        start_level_min: int = 0,
        start_level_max: int = TRAIN_START_LEVEL_MAX,
        enable_jump_per_step: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.max_episode_steps = max_episode_steps
        self._start_level_min = max(0, start_level_min)
        self._start_level_max = min(TRAIN_START_LEVEL_MAX, start_level_max)
        if self._start_level_min > self._start_level_max:
            self._start_level_min = self._start_level_max
        self.enable_jump_per_step = enable_jump_per_step
        self._episode_start_level = 0
        self._forced_start_level: Optional[int] = None
        self._forced_episode_seed: Optional[int] = None
        self.game = IcyTowerGame(seed)
        self.action_space = spaces.Discrete(6)
        high = np.ones(OBS_DIM, dtype=np.float32) * 3.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._screen = None
        self._renderer = None
        self._prev_peak_level = 0
        self._prev_landed_level = 0
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._steps_without_level_gain = 0
        self._jump_reward_this_air = 0.0

    def set_eval_episode(self, start_level: int, episode_seed: int) -> None:
        self._forced_start_level = int(start_level)
        self._forced_episode_seed = int(episode_seed)

    def set_curriculum(
        self,
        start_level_min: int,
        start_level_max: int,
        enable_jump_per_step: bool,
    ) -> None:
        self._start_level_min = max(0, start_level_min)
        self._start_level_max = min(TRAIN_START_LEVEL_MAX, start_level_max)
        if self._start_level_min > self._start_level_max:
            self._start_level_min = self._start_level_max
        self.enable_jump_per_step = enable_jump_per_step

    def _standing_level(self, state) -> Optional[int]:
        if not state.on_ground:
            return None
        if state.standing_platform_id is not None:
            for plat in state.world.platforms:
                if plat.id == state.standing_platform_id:
                    return plat.level
        return int(round(-state.player.feet_y / LEVEL_HEIGHT))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        opts = options or {}
        if self._forced_start_level is not None:
            self._episode_start_level = max(
                0, min(self._forced_start_level, TRAIN_START_LEVEL_MAX)
            )
            episode_seed = int(self._forced_episode_seed or 0)
            self._forced_start_level = None
            self._forced_episode_seed = None
        elif "start_level" in opts:
            self._episode_start_level = max(
                0, min(int(opts["start_level"]), TRAIN_START_LEVEL_MAX)
            )
            episode_seed = (
                int(opts["episode_seed"])
                if "episode_seed" in opts
                else int(self.np_random.integers(0, 2**31 - 1))
            )
        else:
            self._episode_start_level = int(
                self.np_random.integers(
                    self._start_level_min, self._start_level_max + 1
                )
            )
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))

        state = self.game.reset(
            seed=episode_seed,
            start_level=self._episode_start_level,
            win_level=WIN_LEVEL,
        )
        self._prev_peak_level = state.peak_level
        self._prev_landed_level = state.highest_level
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._steps_without_level_gain = 0
        self._jump_reward_this_air = 0.0
        return build_observation(state), self._info(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        move_left = action in (1, 4)
        move_right = action in (2, 5)
        jump = action in (3, 4, 5)

        was_on_ground = self.game.state.on_ground if self.game.state else True

        state = self.game.step(DT, move_left, move_right, jump)
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        left_ground = was_on_ground and not state.on_ground

        if jump and left_ground:
            reward += REWARD_TAKEOFF

        made_progress = False

        peak_gained = state.peak_level - self._prev_peak_level
        if peak_gained > 0:
            reward += REWARD_HIGHEST_LEVEL_BONUS * peak_gained
            self._prev_peak_level = state.peak_level
            made_progress = True

        if state._landed_this_frame:
            landed_gained = state.highest_level - self._prev_landed_level
            if landed_gained > 0:
                for level in range(self._prev_landed_level + 1, state.highest_level + 1):
                    reward += REWARD_LEVEL_UP
                    if level >= HARD_MODE_LEVEL:
                        reward += REWARD_LEVEL_UP_HARD_BONUS
                self._prev_landed_level = state.highest_level
                self._idle_on_start_steps = 0
                made_progress = True

        if made_progress:
            self._steps_without_level_gain = 0
        else:
            self._steps_without_level_gain += 1

        if state.on_ground:
            self._jump_reward_this_air = 0.0
        elif (
            self.enable_jump_per_step
            and state.player.vy < 0
            and self._jump_reward_this_air < MAX_JUMP_REWARD_PER_AIR
        ):
            jump_reward = min(
                REWARD_JUMP_PER_STEP,
                MAX_JUMP_REWARD_PER_AIR - self._jump_reward_this_air,
            )
            reward += jump_reward
            self._jump_reward_this_air += jump_reward

        if state.highest_level <= self._episode_start_level and state.on_ground:
            self._idle_on_start_steps += 1
            if self._idle_on_start_steps > START_LEVEL_IDLE_LIMIT:
                reward -= PENALTY_IDLE_ON_START

        if (
            state.on_ground
            and self._steps_without_level_gain > STAGNATION_STEP_LIMIT
        ):
            reward -= PENALTY_STAGNATION

        margin_bottom = (SCREEN_HEIGHT - (state.player.feet_y - state.camera_y)) / SCREEN_HEIGHT
        if margin_bottom < 0.12:
            reward -= PENALTY_NEAR_BOTTOM

        reward -= PENALTY_PER_STEP

        if state.status == GameStatus.FALLING:
            reward -= PENALTY_DEATH
            terminated = True
        elif state.status == GameStatus.WON:
            reward += REWARD_WIN
            terminated = True

        if self.render_mode == "human":
            self.render()

        return build_observation(state), reward, terminated, truncated, self._info(state)

    def _info(self, state) -> dict:
        p = state.player
        return {
            "highest_level": state.highest_level,
            "peak_level": state.peak_level,
            "start_level": self._episode_start_level,
            "vx": p.vx,
            "vy": p.vy,
            "x": p.x,
            "y": p.y,
            "status": state.status.name,
            "death_reason": state.death_reason,
            "idle_on_start_steps": self._idle_on_start_steps,
            "steps_without_level_gain": self._steps_without_level_gain,
        }

    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        from icy_tower.render import GameRenderer

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Icy Tower RL")
            self._renderer = GameRenderer(self._screen)
        assert self.game.state is not None
        self._renderer.draw(self.game.state)
        pygame.display.flip()
        return None

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None
