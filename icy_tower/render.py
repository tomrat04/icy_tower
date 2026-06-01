from __future__ import annotations

import pygame

from icy_tower.config import (
    COLOR_BG,
    COLOR_DEATH,
    COLOR_PLATFORM,
    COLOR_PLAYER,
    COLOR_TEXT,
    COLOR_WIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WIN_LEVEL,
)
from icy_tower.game import GameState, GameStatus


class GameRenderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 22)
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)

    def draw(self, state: GameState) -> None:
        self.screen.fill(COLOR_BG)
        cam_y = state.camera_y

        for plat in state.world.platforms:
            sy = int(plat.y - cam_y)
            if sy < -50 or sy > SCREEN_HEIGHT + 50:
                continue
            rect = pygame.Rect(int(plat.x), sy, int(plat.width), int(plat.height))
            pygame.draw.rect(self.screen, COLOR_PLATFORM, rect, border_radius=4)

        pr = pygame.Rect(
            int(state.player.x),
            int(state.player.y - cam_y),
            int(state.player.width),
            int(state.player.height),
        )
        color = COLOR_PLAYER if state.status == GameStatus.PLAYING else COLOR_DEATH
        pygame.draw.rect(self.screen, color, pr, border_radius=6)

        hud = f"Poziom: {state.highest_level}/{WIN_LEVEL}"
        self.screen.blit(self.font.render(hud, True, COLOR_TEXT), (12, 12))

        if state.status == GameStatus.WON:
            txt = self.font_big.render("WYGRANA!", True, COLOR_WIN)
            self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        elif state.status == GameStatus.FALLING:
            msg = state.death_reason or "Koniec gry"
            txt = self.font_big.render(msg, True, COLOR_DEATH)
            self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 80)))
