from __future__ import annotations

import random
from typing import Optional

from icy_tower.config import (
    EARLY_PLATFORM_WIDTHS,
    HARD_MODE_LEVEL,
    HARD_PLATFORM_WIDTHS,
    HARD_TWO_PLATFORM_CHANCE,
    JUMP_REACH_ONE_LEVEL,
    LEVEL_HEIGHT,
    MAP_SIDE_MARGIN,
    PLATFORM_HEIGHT,
    PLAYER_WIDTH,
    ROWS_AHEAD,
    SCREEN_WIDTH,
)
from icy_tower.entities import Platform


class WorldGenerator:
    """Proceduralna wieża — każde piętro osiągalne skokiem z niższego."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._next_id = 0
        self.platforms: list[Platform] = []

    def reset(self, start_level: int = 0) -> list[Platform]:
        self._next_id = 0
        self.platforms.clear()
        start_level = max(0, start_level)
        for level in range(start_level, start_level + ROWS_AHEAD + 1):
            self._add_level(level)
        return self.platforms

    def ensure_rows(self, up_to_level: int) -> None:
        """Dopisuje brakujące piętra w górę — nigdy nie usuwa ani nie losuje od nowa."""
        target_max = max(0, up_to_level) + ROWS_AHEAD
        existing = {p.level for p in self.platforms}
        for level in range(0, target_max + 1):
            if level not in existing:
                self._add_level(level)

    def _platforms_on_level(self, level: int) -> list[Platform]:
        return [p for p in self.platforms if p.level == level]

    def _level_layout(self, level: int) -> tuple[int, tuple[float, ...]]:
        if level < HARD_MODE_LEVEL:
            count = 1
            widths_pool = EARLY_PLATFORM_WIDTHS
        elif self.rng.random() < HARD_TWO_PLATFORM_CHANCE:
            count = 2
            widths_pool = HARD_PLATFORM_WIDTHS
        else:
            count = 1
            widths_pool = HARD_PLATFORM_WIDTHS
        widths = tuple(self.rng.choice(widths_pool) for _ in range(count))
        return count, widths

    def _reachable_x_range(
        self, parent: Platform, width: float, reach: float
    ) -> tuple[float, float]:
        """Zakres X nowej platformy, z którego da się wskoczyć z `parent`."""
        lo = parent.x - reach + PLAYER_WIDTH
        hi = parent.x + parent.width + reach - width
        lo = max(MAP_SIDE_MARGIN, lo)
        hi = min(SCREEN_WIDTH - width - MAP_SIDE_MARGIN, hi)
        if lo > hi:
            center = parent.x + parent.width / 2 - width / 2
            center = max(MAP_SIDE_MARGIN, min(center, SCREEN_WIDTH - width - MAP_SIDE_MARGIN))
            return center, center
        return lo, hi

    def _pick_x_above_parents(
        self, parents: list[Platform], width: float, reach: float
    ) -> float:
        if not parents:
            return self.rng.uniform(
                MAP_SIDE_MARGIN, SCREEN_WIDTH - width - MAP_SIDE_MARGIN
            )
        parent = self.rng.choice(parents)
        lo, hi = self._reachable_x_range(parent, width, reach)
        if lo >= hi:
            return lo
        return self.rng.uniform(lo, hi)

    def _add_level(self, level: int) -> None:
        y = -level * LEVEL_HEIGHT
        count, widths = self._level_layout(level)
        parents_below = self._platforms_on_level(level - 1) if level > 0 else []
        reach = JUMP_REACH_ONE_LEVEL

        xs: list[float] = []
        for i, w in enumerate(widths):
            if level == 0 or not parents_below:
                xs.append(
                    self.rng.uniform(MAP_SIDE_MARGIN, SCREEN_WIDTH - w - MAP_SIDE_MARGIN)
                )
            elif i == 0:
                xs.append(self._pick_x_above_parents(parents_below, w, reach))
            else:
                xs.append(self._pick_x_above_parents(parents_below, w, reach))
                xs[-1] = self._separate_x(xs[-1], w, xs[:-1], widths[:i])

        for w, x in zip(widths, xs):
            self.platforms.append(
                Platform(
                    id=self._next_id,
                    x=x,
                    y=y,
                    width=w,
                    height=PLATFORM_HEIGHT,
                    level=level,
                )
            )
            self._next_id += 1

    def _separate_x(
        self,
        x: float,
        width: float,
        other_xs: list[float],
        other_widths: list[float],
        gap: float = MAP_SIDE_MARGIN,
    ) -> float:
        """Minimalna korekta X, żeby platformy na tym samym poziomie się nie nakładały."""
        x = max(MAP_SIDE_MARGIN, min(x, SCREEN_WIDTH - width - MAP_SIDE_MARGIN))
        for ox, ow in zip(other_xs, other_widths):
            if x < ox + ow + gap and x + width + gap > ox:
                x = ox + ow + gap
                x = min(x, SCREEN_WIDTH - width - MAP_SIDE_MARGIN)
        return x

    def get_platforms_near(self, y: float, radius: float = 800.0) -> list[Platform]:
        return [p for p in self.platforms if abs(p.y - y) < radius]
