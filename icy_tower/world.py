from __future__ import annotations

import random
from typing import Optional

from icy_tower.config import (
    LEVEL_HEIGHT,
    MAX_PLATFORMS_PER_LEVEL,
    MIN_PLATFORMS_PER_LEVEL,
    PLATFORM_HEIGHT,
    PLATFORM_WIDTHS,
    ROWS_AHEAD,
    ROWS_BEHIND,
    SCREEN_WIDTH,
)
from icy_tower.entities import Platform


class WorldGenerator:
    """Proceduralna wieża — równe odstępy pionowe, 1–3 platformy na poziom."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._next_id = 0
        self.platforms: list[Platform] = []

    def reset(self, start_level: int = 0) -> list[Platform]:
        self._next_id = 0
        self.platforms.clear()
        for level in range(start_level - ROWS_BEHIND, start_level + ROWS_AHEAD + 1):
            self._add_level(level)
        return self.platforms

    def ensure_rows(self, player_level: int) -> None:
        min_level = player_level - ROWS_BEHIND
        max_level = player_level + ROWS_AHEAD
        existing = {p.level for p in self.platforms}
        for level in range(min_level, max_level + 1):
            if level not in existing:
                self._add_level(level)
        self.platforms = [
            p for p in self.platforms if min_level <= p.level <= max_level
        ]

    def _add_level(self, level: int) -> None:
        y = -level * LEVEL_HEIGHT
        count = self.rng.randint(MIN_PLATFORMS_PER_LEVEL, MAX_PLATFORMS_PER_LEVEL)
        widths = [self.rng.choice(PLATFORM_WIDTHS) for _ in range(count)]
        slots = self._pick_non_overlapping_slots(widths)
        for w, x in zip(widths, slots):
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

    def _pick_non_overlapping_slots(
        self, widths: list[float], margin: float = 24.0
    ) -> list[float]:
        """Losuje pozycje X bez nakładania się platform."""
        attempts = 0
        while attempts < 200:
            xs = []
            ok = True
            for w in widths:
                x = self.rng.uniform(margin, SCREEN_WIDTH - w - margin)
                xs.append(x)
            for i in range(len(widths)):
                for j in range(i + 1, len(widths)):
                    if xs[i] < xs[j] + widths[j] + margin and xs[j] < xs[i] + widths[i] + margin:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return xs
            attempts += 1
        # fallback: równomierny podział
        total = sum(widths) + margin * (len(widths) + 1)
        scale = (SCREEN_WIDTH - margin) / total if total > SCREEN_WIDTH else 1.0
        x = margin
        out = []
        for w in widths:
            ww = w * scale
            out.append(x)
            x += ww + margin
        return out

    def get_platforms_near(self, y: float, radius: float = 800.0) -> list[Platform]:
        return [p for p in self.platforms if abs(p.y - y) < radius]
