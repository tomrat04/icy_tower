from __future__ import annotations

import sys

import pygame

from icy_tower.config import DT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.render import GameRenderer


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen)
    game = IcyTowerGame()
    state = game.reset()

    running = True
    while running:
        move_left = move_right = jump = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = game.reset()

        keys = pygame.key.get_pressed()
        move_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        move_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        jump = keys[pygame.K_SPACE]

        if state.status == GameStatus.PLAYING:
            state = game.step(DT, move_left, move_right, jump)
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
