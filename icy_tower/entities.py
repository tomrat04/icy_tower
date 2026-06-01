from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Platform:
    id: int
    x: float
    y: float
    width: float
    height: float
    level: int

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height

    def contains_x(self, px: float, margin: float = 0.0) -> bool:
        return self.x - margin <= px <= self.x + self.width + margin


@dataclass
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    width: float = 28.0
    height: float = 36.0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def feet_y(self) -> float:
        return self.y + self.height

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height
