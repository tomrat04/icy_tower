from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from icy_tower.config import (
    DT,
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
REWARD_TAKEOFF = 0.5
REWARD_ASCEND_PER_STEP = 0.04
MAX_ASCEND_REWARD_PER_AIR = 0.6
PENALTY_IDLE_ON_START = 0.03
PENALTY_IDLE_ON_LEVEL0 = 0.03
PENALTY_STAGNATION = 0.05
PENALTY_NEAR_BOTTOM = 0.015
PENALTY_PER_STEP = 0.001
PENALTY_DEATH = 25.0
REWARD_WIN = 500.0
START_LEVEL_IDLE_LIMIT = 45
LEVEL0_IDLE_LIMIT = 45
STAGNATION_STEP_LIMIT = 60


class IcyTowerEnv(gym.Env):
    """
    Środowisko Gymnasium dla agenta RL.

    Akcje: 0 — brak, 1 — lewo, 2 — prawo, 3 — skok, 4 — lewo+skok, 5 — prawo+skok.
    Trening: losowy start 0→175, wygrana na 200.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        max_episode_steps: int = 12_000,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.max_episode_steps = max_episode_steps
        self._episode_start_level = 0
        self.game = IcyTowerGame(seed)
        self.action_space = spaces.Discrete(6)
        high = np.ones(OBS_DIM, dtype=np.float32) * 3.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._screen = None
        self._renderer = None
        self._prev_level = 0
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._idle_on_level0_steps = 0
        self._steps_without_level_gain = 0
        self._ascend_reward_this_air = 0.0

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

        self._episode_start_level = int(
            self.np_random.integers(0, TRAIN_START_LEVEL_MAX + 1)
        )
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        state = self.game.reset(
            seed=episode_seed,
            start_level=self._episode_start_level,
            win_level=WIN_LEVEL,
        )
        self._prev_level = state.highest_level
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._idle_on_level0_steps = 0
        self._steps_without_level_gain = 0
        self._ascend_reward_this_air = 0.0
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

        levels_gained = state.highest_level - self._prev_level
        if levels_gained > 0:
            reward += REWARD_LEVEL_UP * levels_gained
            self._prev_level = state.highest_level
            self._idle_on_start_steps = 0
            self._idle_on_level0_steps = 0
            self._steps_without_level_gain = 0
        else:
            self._steps_without_level_gain += 1

        if state.on_ground:
            self._ascend_reward_this_air = 0.0
        elif (
            state.player.vy < 0
            and self._ascend_reward_this_air < MAX_ASCEND_REWARD_PER_AIR
        ):
            ascend = min(
                REWARD_ASCEND_PER_STEP,
                MAX_ASCEND_REWARD_PER_AIR - self._ascend_reward_this_air,
            )
            reward += ascend
            self._ascend_reward_this_air += ascend

        if state.highest_level <= self._episode_start_level and state.on_ground:
            self._idle_on_start_steps += 1
            if self._idle_on_start_steps > START_LEVEL_IDLE_LIMIT:
                reward -= PENALTY_IDLE_ON_START

        standing_level = self._standing_level(state)
        if standing_level == 0:
            self._idle_on_level0_steps += 1
            if self._idle_on_level0_steps > LEVEL0_IDLE_LIMIT:
                reward -= PENALTY_IDLE_ON_LEVEL0
        else:
            self._idle_on_level0_steps = 0

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
            "start_level": self._episode_start_level,
            "vx": p.vx,
            "vy": p.vy,
            "x": p.x,
            "y": p.y,
            "status": state.status.name,
            "death_reason": state.death_reason,
            "idle_on_start_steps": self._idle_on_start_steps,
            "idle_on_level0_steps": self._idle_on_level0_steps,
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
