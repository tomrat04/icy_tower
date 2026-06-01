from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from icy_tower.config import DT, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import OBS_DIM, build_observation


class IcyTowerEnv(gym.Env):
    """
    Środowisko Gymnasium dla agenta RL.

    Akcje: 0 — brak, 1 — lewo (A), 2 — prawo (D).
    Skok / odbicie jest automatyczne po wylądowaniu na platformie.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.game = IcyTowerGame(seed)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self._screen = None
        self._renderer = None
        self._prev_level = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        state = self.game.reset(seed=self._seed)
        self._prev_level = 0
        return build_observation(state), self._info(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        move_left = action == 1
        move_right = action == 2
        state = self.game.step(DT, move_left, move_right)

        reward = 0.0
        terminated = False
        truncated = False

        if state.highest_level > self._prev_level:
            reward += 1.0
            self._prev_level = state.highest_level

        reward += max(0.0, -state.player.vy) * 0.0001

        if state.status == GameStatus.FALLING:
            reward -= 50.0
            terminated = True
        elif state.status == GameStatus.WON:
            reward += 200.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return build_observation(state), reward, terminated, truncated, self._info(state)

    def _info(self, state) -> dict:
        return {
            "highest_level": state.highest_level,
            "status": state.status.name,
            "death_reason": state.death_reason,
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
