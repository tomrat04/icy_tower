from __future__ import annotations

import pygame

from icy_tower.config import (
    COLOR_BG,
    COLOR_COMBO_BAR,
    COLOR_COMBO_BAR_BG,
    COLOR_DEATH,
    COLOR_PLATFORM,
    COLOR_PLAYER,
    COLOR_TEXT,
    COLOR_WIN,
    COLOR_RUN_BAR,
    COMBO_TIMER_DURATION,
    RUN_MOMENTUM_FOR_MAX_JUMP,
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

        hud = (
            f"Poziom: {state.highest_level}/{WIN_LEVEL}  "
            f"Wynik: {state.score}"
        )
        self.screen.blit(self.font.render(hud, True, COLOR_TEXT), (12, 12))

        if state.status == GameStatus.PLAYING:
            self._draw_run_momentum_bar(state)

            if state.combo_chain > 0 or state.combo_timer > 0:
                self._draw_combo_bar(state)

            if state.wall_chain > 0:
                wall = self.font.render(
                    f"Odbicie ściany {state.wall_chain}/2",
                    True,
                    COLOR_RUN_BAR,
                )
                self.screen.blit(wall, (240, 36))

            if state.last_combo_score > 0:
                flash = self.font.render(
                    f"+{state.last_combo_score} combo!",
                    True,
                    COLOR_COMBO_BAR,
                )
                self.screen.blit(flash, (12, 84))

        if state.status == GameStatus.WON:
            txt = self.font_big.render("WYGRANA!", True, COLOR_WIN)
            self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        elif state.status == GameStatus.FALLING:
            msg = state.death_reason or "Koniec gry"
            txt = self.font_big.render(msg, True, COLOR_DEATH)
            self.screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2, 80)))

    def _draw_run_momentum_bar(self, state: GameState) -> None:
        bar_w = 180
        bar_h = 12
        x, y = 12, 36
        cap = RUN_MOMENTUM_FOR_MAX_JUMP if RUN_MOMENTUM_FOR_MAX_JUMP > 0 else 1.0
        fill = max(0.0, min(1.0, state.run_momentum / cap))
        pygame.draw.rect(self.screen, COLOR_COMBO_BAR_BG, (x, y, bar_w, bar_h), border_radius=3)
        if fill > 0:
            pygame.draw.rect(
                self.screen,
                COLOR_RUN_BAR,
                (x, y, int(bar_w * fill), bar_h),
                border_radius=3,
            )
        pct = int(fill * 100)
        label = self.font.render(f"Pęd pionowy {pct}%", True, COLOR_RUN_BAR)
        self.screen.blit(label, (x + bar_w + 8, y - 2))

    def _draw_combo_bar(self, state: GameState) -> None:
        bar_w = 220
        bar_h = 14
        x, y = 12, 58
        fill = 0.0
        if COMBO_TIMER_DURATION > 0:
            fill = max(0.0, min(1.0, state.combo_timer / COMBO_TIMER_DURATION))
        pygame.draw.rect(self.screen, COLOR_COMBO_BAR_BG, (x, y, bar_w, bar_h), border_radius=4)
        if fill > 0:
            pygame.draw.rect(
                self.screen,
                COLOR_COMBO_BAR,
                (x, y, int(bar_w * fill), bar_h),
                border_radius=4,
            )
        label = self.font.render(f"COMBO x{state.combo_chain}", True, COLOR_COMBO_BAR)
        self.screen.blit(label, (x + bar_w + 8, y - 2))
