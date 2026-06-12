"""Test agenta — normalne tempo 60 FPS jak w play.py (nie turbo jak env.step)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from icy_tower.config import DT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.env import IcyTowerEnv
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import build_observation
from icy_tower.render import GameRenderer
from icy_tower.wrappers import IcyTowerVecWrapper


def _find_vecnormalize(model_path: Path) -> Path | None:
    candidates = [
        model_path.parent / "vecnormalize.pkl",
        model_path.parent / "best" / "vecnormalize.pkl",
        model_path.parent.parent / "vecnormalize.pkl",
        Path("models/best/vecnormalize.pkl"),
        Path("models/vecnormalize.pkl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_test(
    model_path: str = "models/best/best_model.zip",
    vecnormalize_path: str | None = None,
    fps: int = FPS,
    pause_on_end: float = 2.0,
    seed: int | None = None,
) -> None:
    path = Path(model_path)
    if not path.suffix:
        path = Path(str(path) + ".zip")
    if not path.exists():
        raise FileNotFoundError(f"Brak modelu: {path}")

    vec_path = Path(vecnormalize_path) if vecnormalize_path else _find_vecnormalize(path)
    vecnorm: VecNormalize | None = None
    if vec_path is not None:
        dummy = DummyVecEnv(
            [lambda: IcyTowerVecWrapper(IcyTowerEnv(seed=seed or 0))]
        )
        vecnorm = VecNormalize.load(str(vec_path), dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False

    model = PPO.load(path)
    game = IcyTowerGame(seed=seed)
    state = game.reset(seed=seed)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Icy Tower — test agenta")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen)

    running = True
    pause_until = 0.0
    episode = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = game.reset(seed=seed)
                    pause_until = 0.0
                    episode += 1

        if time.monotonic() < pause_until:
            renderer.draw(state)
            pygame.display.flip()
            clock.tick(fps)
            continue

        if state.status == GameStatus.PLAYING:
            obs = build_observation(state)
            if vecnorm is not None:
                obs = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            move_left = action in (1, 4)
            move_right = action in (2, 5)
            jump = action in (3, 4, 5)
            state = game.step(DT, move_left, move_right, jump)
        elif state.status == GameStatus.FALLING:
            state = game.step(DT, False, False)
            if state.player.y - state.camera_y > SCREEN_HEIGHT + 120:
                print(
                    f"Epizod {episode}: poziom {state.highest_level}, "
                    f"{state.death_reason or 'koniec'}"
                )
                pause_until = time.monotonic() + pause_on_end
                state = game.reset(seed=seed)
                episode += 1
        elif state.status == GameStatus.WON:
            print(f"Epizod {episode}: WYGRANA na poziomie {state.highest_level}")
            pause_until = time.monotonic() + pause_on_end
            state = game.reset(seed=seed)
            episode += 1

        renderer.draw(state)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    run_test()
