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

REWARD_LEVEL_UP = 10.0
REWARD_LEVEL_UP_HARD_BONUS = 5.0
REWARD_HIGHEST_LEVEL_BONUS = 0.2
REWARD_TAKEOFF = 0.2
REWARD_JUMP_PER_STEP = 0.05

PENALTY_STAGNATION = 0.05
PENALTY_NEAR_BOTTOM = 0.02
PENALTY_PER_STEP = 0.001
PENALTY_DEATH = 25.0
REWARD_WIN = 200.0
STAGNATION_STEP_LIMIT = 120


class IcyTowerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None, max_episode_steps: int = 12_000, start_level_min: int = 0, start_level_max: int = TRAIN_START_LEVEL_MAX, enable_jump_per_step: bool = True,):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.max_episode_steps = max_episode_steps
        self._start_level_min = start_level_min
        self._start_level_max = start_level_max
        self.enable_jump_per_step = enable_jump_per_step
        self._episode_start_level = 0
        self._forced_start_level = None
        self.game = IcyTowerGame(seed)
        self.action_space = spaces.Discrete(6)
        high = np.ones(OBS_DIM, dtype=np.float32) * 1.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._screen = None
        self._renderer = None
        self._prev_peak_level = 0
        self._prev_landed_level = 0
        self._step_count = 0
        self._steps_without_level_gain = 0

    def set_eval_episode(self, start_level):
        self._forced_start_level = int(start_level)

    def set_curriculum(self, start_level_min, start_level_max, enable_jump_per_step):
        self._start_level_min = start_level_min
        self._start_level_max = start_level_max
        self.enable_jump_per_step = enable_jump_per_step

    def standing_level(self, state):
        if not state.on_ground:
            return None
        if state.standing_platform_id is not None:
            for plat in state.world.platforms:
                if plat.id == state.standing_platform_id:
                    return plat.level
        return None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        opts = options or {}
        if self._forced_start_level is not None:
            self._episode_start_level = max(0, min(self._forced_start_level, TRAIN_START_LEVEL_MAX))
            episode_seed = 0
            self._forced_start_level = None
        elif "start_level" in opts:
            self._episode_start_level = max(0, min(int(opts["start_level"]), TRAIN_START_LEVEL_MAX))
            episode_seed = int(opts["episode_seed"]) if "episode_seed" in opts else int(
                self.np_random.integers(0, 2 ** 31 - 1))
        else:
            self._episode_start_level = int(self.np_random.integers(self._start_level_min, self._start_level_max + 1))
            episode_seed = int(self.np_random.integers(0, 2 ** 31 - 1))

        state = self.game.reset(seed=episode_seed, start_level=self._episode_start_level, win_level=WIN_LEVEL)
        self._prev_peak_level = state.peak_level
        self._prev_landed_level = state.highest_level
        self._step_count = 0
        self._steps_without_level_gain = 0

        return build_observation(state), self._info(state)

    def step(self, action):
        move_left = False
        move_right = False
        jump = False

        if action == 1:
            move_left = True
        elif action == 2:
            move_right = True
        elif action == 3:
            jump = True
        elif action == 4:
            move_left = True
            jump = True
        elif action == 5:
            move_right = True
            jump = True

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
                made_progress = True

        if made_progress:
            self._steps_without_level_gain = 0
        else:
            self._steps_without_level_gain += 1

        if not state.on_ground and self.enable_jump_per_step and state.player.vy < 0:
            reward += REWARD_JUMP_PER_STEP

        reward -= PENALTY_PER_STEP

        if state.on_ground and self._steps_without_level_gain > STAGNATION_STEP_LIMIT:
            reward -= PENALTY_STAGNATION

        margin_bottom = (SCREEN_HEIGHT - (state.player.feet_y - state.camera_y)) / SCREEN_HEIGHT
        if margin_bottom < 0.15:
            reward -= PENALTY_NEAR_BOTTOM

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
            pygame.display.set_caption("Icy Tower")
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