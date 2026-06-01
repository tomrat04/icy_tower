from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from icy_tower.config import (
    BOUNCE_VELOCITY,
    FALL_DEATH_MARGIN,
    GRAVITY,
    LEVEL_HEIGHT,
    MAX_BOUNCES_PER_PLATFORM,
    MAX_FALL_SPEED,
    MOVE_SPEED,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    START_BOUNCE_VELOCITY,
    WIN_LEVEL,
)
from icy_tower.entities import Platform, Player
from icy_tower.world import WorldGenerator


class GameStatus(Enum):
    PLAYING = auto()
    FALLING = auto()
    WON = auto()


@dataclass
class GameState:
    player: Player
    world: WorldGenerator
    status: GameStatus = GameStatus.PLAYING
    highest_level: int = 0
    reference_platform_y: Optional[float] = None
    reference_platform_level: Optional[int] = None
    platform_bounce_counts: dict[int, int] = field(default_factory=dict)
    death_reason: str = ""
    move_left: bool = False
    move_right: bool = False
    time: float = 0.0
    _landed_this_frame: bool = field(default=False, repr=False)

    @property
    def camera_y(self) -> float:
        """Kamera śledzi gracza — gracz w dolnej 1/3 ekranu."""
        return self.player.y - SCREEN_HEIGHT * 0.65


class IcyTowerGame:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.state: Optional[GameState] = None

    def reset(self, seed: Optional[int] = None) -> GameState:
        if seed is not None:
            self.seed = seed
        world = WorldGenerator(self.seed)
        world.reset(start_level=0)
        start_plat = min(world.platforms, key=lambda p: p.level)
        px = start_plat.x + start_plat.width / 2 - PLAYER_WIDTH / 2
        py = start_plat.y - PLAYER_HEIGHT
        player = Player(x=px, y=py, vy=START_BOUNCE_VELOCITY)
        self.state = GameState(
            player=player,
            world=world,
            highest_level=start_plat.level,
            reference_platform_y=start_plat.y,
            reference_platform_level=start_plat.level,
            platform_bounce_counts={},
        )
        return self.state

    def step(self, dt: float, move_left: bool, move_right: bool) -> GameState:
        s = self.state
        if s is None:
            raise RuntimeError("Wywołaj reset() przed step().")
        if s.status == GameStatus.FALLING:
            s.player.vy = min(s.player.vy + GRAVITY * dt, MAX_FALL_SPEED * 1.2)
            s.player.x += s.player.vx * dt
            s.player.y += s.player.vy * dt
            return s

        if s.status == GameStatus.WON:
            return s

        s.move_left = move_left
        s.move_right = move_right
        s.time += dt
        s._landed_this_frame = False

        if move_left and not move_right:
            s.player.vx = -MOVE_SPEED
        elif move_right and not move_left:
            s.player.vx = MOVE_SPEED
        else:
            s.player.vx = 0.0

        s.player.vy = min(s.player.vy + GRAVITY * dt, MAX_FALL_SPEED)
        s.player.x += s.player.vx * dt
        s.player.y += s.player.vy * dt

        s.player.x = max(0.0, min(s.player.x, SCREEN_WIDTH - PLAYER_WIDTH))

        if (
            s.reference_platform_y is not None
            and s.player.vy > 0
            and s.player.feet_y > s.reference_platform_y + FALL_DEATH_MARGIN
        ):
            self._trigger_fall(s, "Spadłeś zbyt nisko — poniżej platformy startowej skoku.")
            return s

        if s.player.vy >= 0:
            self._resolve_platform_landings(s, dt)

        player_level = self._level_from_y(s.player.y)
        s.highest_level = max(s.highest_level, player_level)
        s.world.ensure_rows(player_level)

        if s.highest_level >= WIN_LEVEL:
            s.status = GameStatus.WON

        return s

    def _level_from_y(self, y: float) -> int:
        return int(round(-y / LEVEL_HEIGHT))

    def _resolve_platform_landings(self, s: GameState, dt: float) -> None:
        p = s.player
        prev_feet = p.feet_y - p.vy * dt
        candidates = sorted(
            s.world.get_platforms_near(p.y, radius=LEVEL_HEIGHT * 2),
            key=lambda pl: pl.y,
            reverse=True,
        )
        for plat in candidates:
            top = plat.y
            if p.vy < 0:
                continue
            if prev_feet <= top <= p.feet_y + 2:
                if not plat.contains_x(p.center_x, margin=2):
                    continue
                self._land_on_platform(s, plat)
                break

    def _land_on_platform(self, s: GameState, plat: Platform) -> None:
        p = s.player
        p.y = plat.y - PLAYER_HEIGHT
        p.vy = BOUNCE_VELOCITY
        s._landed_this_frame = True

        bounces = s.platform_bounce_counts.get(plat.id, 0) + 1
        s.platform_bounce_counts[plat.id] = bounces
        if bounces > MAX_BOUNCES_PER_PLATFORM:
            self._trigger_fall(
                s,
                f"Za dużo odbić z tej platformy ({MAX_BOUNCES_PER_PLATFORM} max).",
            )
            return

        s.reference_platform_y = plat.y
        s.reference_platform_level = plat.level
        s.highest_level = max(s.highest_level, plat.level)

    def _trigger_fall(self, s: GameState, reason: str) -> None:
        s.status = GameStatus.FALLING
        s.death_reason = reason
        s.player.vy = min(s.player.vy, 0) + 200

    def visible_platforms(self) -> list[Platform]:
        if self.state is None:
            return []
        cam = self.state.camera_y
        return [
            p
            for p in self.state.world.platforms
            if cam - 50 < p.y < cam + SCREEN_HEIGHT + 50
        ]
