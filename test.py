"""Test agenta — normalne tempo 60 FPS jak w play.py (nie turbo jak env.step)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO

from icy_tower.config import DT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import build_observation
from icy_tower.render import GameRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Podgląd wytrenowanego agenta")
    parser.add_argument("--model", type=str, default="models/icy_ppo_final.zip")
    parser.add_argument("--fps", type=int, default=FPS, help="Tempo renderowania")
    parser.add_argument(
        "--pause-on-end",
        type=float,
        default=2.0,
        help="Pauza (s) po śmierci/wygranej przed restartem",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.model)
    if not path.suffix:
        path = Path(str(path) + ".zip")
    if not path.exists():
        print(f"Brak modelu: {path}", file=sys.stderr)
        sys.exit(1)

    model = PPO.load(path)
    game = IcyTowerGame(seed=args.seed)
    state = game.reset(seed=args.seed)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Icy Tower — test agenta")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen)

    running = True
    pause_until = 0.0
    episode = 1

    while running:
        move_left = move_right = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = game.reset(seed=args.seed)
                    pause_until = 0.0
                    episode += 1

        if time.monotonic() < pause_until:
            renderer.draw(state)
            pygame.display.flip()
            clock.tick(args.fps)
            continue

        if state.status == GameStatus.PLAYING:
            obs = build_observation(state)
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
                pause_until = time.monotonic() + args.pause_on_end
                state = game.reset(seed=args.seed)
                episode += 1
        elif state.status == GameStatus.WON:
            print(f"Epizod {episode}: WYGRANA na poziomie {state.highest_level}")
            pause_until = time.monotonic() + args.pause_on_end
            state = game.reset(seed=args.seed)
            episode += 1

        renderer.draw(state)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
