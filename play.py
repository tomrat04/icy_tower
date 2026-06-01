"""Gra ręczna — sterowanie A/D, ESC wyjście, R restart."""

from __future__ import annotations

import sys

import pygame

from icy_tower.config import DT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.render import GameRenderer


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Icy Tower — A/D")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen)
    game = IcyTowerGame()
    state = game.reset()

    running = True
    while running:
        move_left = move_right = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = game.reset()

        keys = pygame.key.get_pressed()
        move_left = keys[pygame.K_a]
        move_right = keys[pygame.K_d]

        if state.status == GameStatus.PLAYING:
            state = game.step(DT, move_left, move_right)
        elif state.status == GameStatus.FALLING:
            state = game.step(DT, False, False)
            if state.player.y - state.camera_y > SCREEN_HEIGHT + 120:
                state = game.reset()
        elif state.status == GameStatus.WON:
            pass

        renderer.draw(state)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
