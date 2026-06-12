from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from icy_tower.config import (
    AUTO_SCROLL_SPEED,
    CAMERA_PLAYER_RATIO,
    CAMERA_SMOOTH,
    CAMERA_SMOOTH_AIR,
    GROUND_DECEL,
    GROUND_STOP_SPEED,
    GRAVITY,
    JUMP_VELOCITY_STAND,
    LEVEL_HEIGHT,
    MAX_FALL_SPEED,
    MOMENTUM_JUMP_BOOST,
    MOVE_SPEED,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    ROWS_AHEAD,
    RUN_MOMENTUM_BUILD,
    RUN_MOMENTUM_DECAY,
    RUN_MOMENTUM_FOR_MAX_JUMP,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SCROLL_DEATH_MARGIN,
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
    peak_level: int = 0
    win_level: int = WIN_LEVEL
    camera_y: float = 0.0
    camera_pressure_y: float = 0.0
    on_ground: bool = False
    run_momentum: float = 0.0
    death_reason: str = ""
    move_left: bool = False
    move_right: bool = False
    time: float = 0.0
    land_grace: float = 0.0
    standing_platform_id: Optional[int] = None
    _landed_this_frame: bool = field(default=False, repr=False)


class IcyTowerGame:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.state: Optional[GameState] = None
        self._win_level = WIN_LEVEL

    def reset(
        self,
        seed: Optional[int] = None,
        start_level: int = 0,
        win_level: Optional[int] = None,
    ) -> GameState:
        if seed is not None:
            self.seed = seed
        self._win_level = win_level if win_level is not None else WIN_LEVEL
        start_level = max(0, start_level)

        world = WorldGenerator(self.seed)
        world.reset(start_level=0)
        world.ensure_rows(start_level + ROWS_AHEAD)

        plats = [p for p in world.platforms if p.level == start_level]
        if not plats:
            raise RuntimeError(f"Brak platformy na poziomie {start_level}")
        start_plat = world.rng.choice(plats)

        px = start_plat.x + start_plat.width / 2 - PLAYER_WIDTH / 2
        py = start_plat.y - PLAYER_HEIGHT
        player = Player(x=px, y=py, vy=0.0)
        cam_y = py - SCREEN_HEIGHT * CAMERA_PLAYER_RATIO
        self.state = GameState(
            player=player,
            world=world,
            highest_level=start_level,
            peak_level=start_level,
            win_level=self._win_level,
            camera_y=cam_y,
            camera_pressure_y=cam_y,
            on_ground=True,
            standing_platform_id=start_plat.id,
        )
        return self.state

    def step(
        self, dt: float, move_left: bool, move_right: bool, jump: bool = False
    ) -> GameState:
        s = self.state
        if s is None:
            raise RuntimeError("Wywołaj reset() przed step().")
        if s.status == GameStatus.FALLING:
            s.player.vy = min(s.player.vy + GRAVITY * dt, MAX_FALL_SPEED * 1.2)
            s.player.x += s.player.vx * dt
            s.player.y += s.player.vy * dt
            self._update_camera(s, dt)
            return s

        if s.status == GameStatus.WON:
            return s

        s.move_left = move_left
        s.move_right = move_right
        s.time += dt
        s._landed_this_frame = False

        if s.on_ground:
            self._update_run_momentum(s, dt, move_left, move_right)
            self._apply_ground_friction(s, dt, move_left, move_right)
        else:
            s.run_momentum = max(0.0, s.run_momentum - RUN_MOMENTUM_DECAY * 0.35 * dt)

        self._apply_horizontal_input(s, move_left, move_right)

        if s.on_ground and s.standing_platform_id is not None:
            self._snap_to_standing_platform(s)
            s.player.vy = 0.0
        else:
            s.player.vy = min(s.player.vy + GRAVITY * dt, MAX_FALL_SPEED)
            s.player.y += s.player.vy * dt

        s.player.x += s.player.vx * dt

        self._clamp_to_walls(s)

        if not s.on_ground and s.player.vy >= 0:
            self._resolve_platform_landings(s, dt)

        if (
            s.on_ground
            and not s._landed_this_frame
            and s.land_grace <= 0.0
            and not self._is_on_standing_platform(s)
        ):
            s.on_ground = False
            s.standing_platform_id = None

        if s.land_grace > 0.0:
            s.land_grace = max(0.0, s.land_grace - dt)

        if jump and s.on_ground and not s._landed_this_frame:
            self._do_jump(s)

        self._update_camera(s, dt)

        screen_y = s.player.feet_y - s.camera_y
        if screen_y > SCREEN_HEIGHT + SCROLL_DEATH_MARGIN:
            self._trigger_fall(s, "Wypadłeś poza dolną krawędź ekranu.")
            return s

        player_level = self._level_from_y(s.player.y)
        s.peak_level = max(s.peak_level, player_level)
        view_level = max(s.highest_level, s.peak_level)
        s.world.ensure_rows(view_level)

        if s.highest_level >= s.win_level:
            s.status = GameStatus.WON

        return s

    def _level_from_y(self, y: float) -> int:
        return int(round(-y / LEVEL_HEIGHT))

    def _momentum_ratio(self, momentum: float) -> float:
        if RUN_MOMENTUM_FOR_MAX_JUMP <= 0:
            return 0.0
        return min(1.0, momentum / RUN_MOMENTUM_FOR_MAX_JUMP)

    def _vertical_from_momentum(self, momentum: float, extra: float = 1.0) -> float:
        t = self._momentum_ratio(momentum)
        return JUMP_VELOCITY_STAND * (1.0 + MOMENTUM_JUMP_BOOST * t) * extra

    def _update_run_momentum(
        self, s: GameState, dt: float, move_left: bool, move_right: bool
    ) -> None:
        if move_left or move_right:
            s.run_momentum = min(
                RUN_MOMENTUM_FOR_MAX_JUMP,
                s.run_momentum + RUN_MOMENTUM_BUILD * dt,
            )
        else:
            s.run_momentum = max(0.0, s.run_momentum - RUN_MOMENTUM_DECAY * dt)

    def _do_jump(self, s: GameState) -> None:
        p = s.player
        p.vy = self._vertical_from_momentum(s.run_momentum)
        s.on_ground = False
        s.standing_platform_id = None
        s._landed_this_frame = False

    def _apply_horizontal_input(
        self, s: GameState, move_left: bool, move_right: bool
    ) -> None:
        p = s.player
        if move_left and not move_right:
            p.vx = -MOVE_SPEED
        elif move_right and not move_left:
            p.vx = MOVE_SPEED

    def _apply_ground_friction(
        self, s: GameState, dt: float, move_left: bool, move_right: bool
    ) -> None:
        if move_left or move_right:
            return
        p = s.player
        if abs(p.vx) <= GROUND_STOP_SPEED:
            p.vx = 0.0
            return
        decel = GROUND_DECEL * dt
        if p.vx > 0:
            p.vx = max(0.0, p.vx - decel)
        else:
            p.vx = min(0.0, p.vx + decel)

    def _clamp_to_walls(self, s: GameState) -> None:
        p = s.player
        if p.x <= 0.0:
            p.x = 0.0
            if p.vx < 0.0:
                p.vx = 0.0
        elif p.x >= SCREEN_WIDTH - PLAYER_WIDTH:
            p.x = SCREEN_WIDTH - PLAYER_WIDTH
            if p.vx > 0.0:
                p.vx = 0.0

    def _update_camera(self, s: GameState, dt: float) -> None:
        follow = s.player.y - SCREEN_HEIGHT * CAMERA_PLAYER_RATIO
        s.camera_pressure_y = min(s.camera_pressure_y, follow + 30.0)
        s.camera_pressure_y -= AUTO_SCROLL_SPEED * dt
        target = min(follow, s.camera_pressure_y)
        smooth = CAMERA_SMOOTH if s.on_ground else CAMERA_SMOOTH_AIR
        blend = min(1.0, smooth * dt)
        s.camera_y += (target - s.camera_y) * blend

    def _get_platform_by_id(self, s: GameState, plat_id: int) -> Optional[Platform]:
        for plat in s.world.platforms:
            if plat.id == plat_id:
                return plat
        return None

    def _is_on_standing_platform(self, s: GameState) -> bool:
        if s.standing_platform_id is None:
            return False
        plat = self._get_platform_by_id(s, s.standing_platform_id)
        if plat is None:
            return False
        p = s.player
        if abs(p.feet_y - plat.y) > 12:
            return False
        return self._overlaps_platform_x(p, plat)

    def _overlaps_platform_x(self, p: Player, plat: Platform) -> bool:
        return p.x + p.width > plat.x + 2 and p.x < plat.x + plat.width - 2

    def _snap_to_standing_platform(self, s: GameState) -> None:
        plat = self._get_platform_by_id(s, s.standing_platform_id) if s.standing_platform_id else None
        if plat is None:
            return
        p = s.player
        p.y = plat.y - PLAYER_HEIGHT
        if not self._overlaps_platform_x(p, plat):
            s.on_ground = False
            s.standing_platform_id = None

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
            if prev_feet <= top <= p.feet_y + 4:
                if not self._overlaps_platform_x(p, plat):
                    continue
                self._land_on_platform(s, plat)
                break

    def _land_on_platform(self, s: GameState, plat: Platform) -> None:
        p = s.player
        p.y = plat.y - PLAYER_HEIGHT
        p.vy = 0.0
        s.on_ground = True
        s.standing_platform_id = plat.id
        s._landed_this_frame = True
        s.land_grace = 0.35
        s.highest_level = max(s.highest_level, plat.level)

    def _trigger_fall(self, s: GameState, reason: str) -> None:
        s.status = GameStatus.FALLING
        s.death_reason = reason
        s.on_ground = False
        s.standing_platform_id = None
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
