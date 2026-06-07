from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from icy_tower.config import DT, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import OBS_DIM, build_observation

# --- Nagrody i kary (na krok, o ile nie podano inaczej) ---
REWARD_LEVEL_UP = 5.0
REWARD_COMBO_SCORE_FACTOR = 0.05
REWARD_JUMP = 0.15
REWARD_ASCEND = 0.003
PENALTY_IDLE_ON_START = 0.02
PENALTY_WALL_SPAM = 0.08
PENALTY_NEAR_BOTTOM = 0.015
PENALTY_PER_STEP = 0.001
PENALTY_DEATH = 40.0
REWARD_WIN = 500.0
START_LEVEL_IDLE_LIMIT = 400


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
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.max_episode_steps = max_episode_steps
        self.game = IcyTowerGame(seed)
        self.action_space = spaces.Discrete(6)
        high = np.ones(OBS_DIM, dtype=np.float32) * 3.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._screen = None
        self._renderer = None
        self._prev_level = 0
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._was_in_air = False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        state = self.game.reset(seed=self._seed)
        self._prev_level = state.highest_level
        self._step_count = 0
        self._idle_on_start_steps = 0
        self._was_in_air = False
        return build_observation(state), self._info(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        move_left = action in (1, 4)
        move_right = action in (2, 5)
        jump = action in (3, 4, 5)
        state = self.game.step(DT, move_left, move_right, jump)
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        levels_gained = state.highest_level - self._prev_level
        if levels_gained > 0:
            reward += REWARD_LEVEL_UP * levels_gained
            self._prev_level = state.highest_level
            self._idle_on_start_steps = 0

        if state.last_combo_score > 0:
            reward += state.last_combo_score * REWARD_COMBO_SCORE_FACTOR

        if not state.on_ground:
            self._was_in_air = True
            if state.player.vy < 0:
                reward += REWARD_JUMP
            reward += REWARD_ASCEND

        if state.highest_level <= 0 and state.on_ground:
            self._idle_on_start_steps += 1
            if self._idle_on_start_steps > START_LEVEL_IDLE_LIMIT:
                reward -= PENALTY_IDLE_ON_START

        if state.wall_chain >= 2 and not state.on_ground:
            reward -= PENALTY_WALL_SPAM

        if jump and state.wall_chain == 1 and not state.on_ground:
            reward += 0.4

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
            "score": state.score,
            "combo_chain": state.combo_chain,
            "vx": p.vx,
            "vy": p.vy,
            "x": p.x,
            "y": p.y,
            "status": state.status.name,
            "death_reason": state.death_reason,
            "idle_on_start_steps": self._idle_on_start_steps,
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
