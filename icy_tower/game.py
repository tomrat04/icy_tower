from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from icy_tower.config import (
    AUTO_SCROLL_SPEED,
    CAMERA_PLAYER_RATIO,
    CAMERA_SMOOTH,
    CAMERA_SMOOTH_AIR,
    COMBO_BASE_SCORE,
    COMBO_MIN_LEVELS_GAIN,
    COMBO_TIMER_DECAY,
    COMBO_TIMER_DURATION,
    GROUND_DECEL,
    GROUND_STOP_SPEED,
    GRAVITY,
    JUMP_VELOCITY_MAX,
    JUMP_VELOCITY_STAND,
    LEVEL_HEIGHT,
    MAX_FALL_SPEED,
    MAX_WALL_BOUNCES_PER_AIR,
    MOVE_SPEED,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    RUN_MOMENTUM_BUILD,
    RUN_MOMENTUM_DECAY,
    RUN_MOMENTUM_FOR_MAX_JUMP,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SCROLL_DEATH_MARGIN,
    WALL_JUMP_MARGIN,
    WALL_JUMP_MIN_MOMENTUM,
    WALL_VY_BOOST,
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
    camera_y: float = 0.0
    camera_pressure_y: float = 0.0
    on_ground: bool = False
    run_momentum: float = 0.0
    wall_chain: int = 0
    score: int = 0
    combo_chain: int = 0
    combo_timer: float = 0.0
    last_combo_score: int = 0
    jump_takeoff_level: Optional[int] = None
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

    def reset(self, seed: Optional[int] = None) -> GameState:
        if seed is not None:
            self.seed = seed
        world = WorldGenerator(self.seed)
        world.reset(start_level=0)
        start_plat = min(world.platforms, key=lambda p: p.level)
        px = start_plat.x + start_plat.width / 2 - PLAYER_WIDTH / 2
        py = start_plat.y - PLAYER_HEIGHT
        player = Player(x=px, y=py, vy=0.0)
        cam_y = py - SCREEN_HEIGHT * CAMERA_PLAYER_RATIO
        self.state = GameState(
            player=player,
            world=world,
            highest_level=start_plat.level,
            camera_y=cam_y,
            camera_pressure_y=cam_y,
            on_ground=True,
            standing_platform_id=start_plat.id,
            jump_takeoff_level=start_plat.level,
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
        s.last_combo_score = 0

        self._tick_combo_timer(s, dt)

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

        if jump:
            if not self._try_wall_jump(s, move_left, move_right):
                if s.on_ground and not s._landed_this_frame:
                    self._do_jump(s)

        self._update_camera(s, dt)

        screen_y = s.player.feet_y - s.camera_y
        if screen_y > SCREEN_HEIGHT + SCROLL_DEATH_MARGIN:
            self._trigger_fall(s, "Wypadłeś poza dolną krawędź ekranu.")
            return s

        player_level = self._level_from_y(s.player.y)
        s.highest_level = max(s.highest_level, player_level)
        s.world.ensure_rows(s.highest_level)

        if s.highest_level >= WIN_LEVEL:
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
        vy = JUMP_VELOCITY_STAND + (JUMP_VELOCITY_MAX - JUMP_VELOCITY_STAND) * t
        return vy * extra

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

    def _stand_level(self, s: GameState) -> int:
        p = s.player
        feet = p.feet_y
        best: Optional[int] = None
        for plat in s.world.get_platforms_near(p.y, radius=LEVEL_HEIGHT):
            if abs(feet - plat.y) > 4:
                continue
            if plat.contains_x(p.center_x, margin=2):
                if best is None or plat.level > best:
                    best = plat.level
        return best if best is not None else self._level_from_y(p.y)

    def _do_jump(self, s: GameState) -> None:
        p = s.player
        p.vy = self._vertical_from_momentum(s.run_momentum)
        s.jump_takeoff_level = self._stand_level(s)
        s.on_ground = False
        s.standing_platform_id = None
        s._landed_this_frame = False
        s.wall_chain = 0

    def _try_wall_jump(
        self, s: GameState, move_left: bool, move_right: bool
    ) -> bool:
        """Wall jump: przy ścianie + skok odbija w drugą stronę (max 2× w locie)."""
        p = s.player
        if s.wall_chain >= MAX_WALL_BOUNCES_PER_AIR:
            return False

        at_left = p.x <= WALL_JUMP_MARGIN
        at_right = p.x >= SCREEN_WIDTH - PLAYER_WIDTH - WALL_JUMP_MARGIN
        if not at_left and not at_right:
            return False

        momentum = max(s.run_momentum, WALL_JUMP_MIN_MOMENTUM)
        if s.wall_chain == 0 and s.on_ground and s.run_momentum < WALL_JUMP_MIN_MOMENTUM:
            return False

        if at_left:
            p.vx = MOVE_SPEED
            p.x = 0.0
        else:
            p.vx = -MOVE_SPEED
            p.x = SCREEN_WIDTH - PLAYER_WIDTH

        s.wall_chain += 1
        if s.wall_chain == 1:
            base_vy = self._vertical_from_momentum(momentum, extra=WALL_VY_BOOST)
            if p.vy > base_vy:
                p.vy = base_vy

        s.on_ground = False
        s.standing_platform_id = None
        s._landed_this_frame = False
        return True

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

    def _tick_combo_timer(self, s: GameState, dt: float) -> None:
        if s.combo_timer <= 0.0:
            return
        s.combo_timer -= dt * COMBO_TIMER_DECAY
        if s.combo_timer <= 0.0:
            self._end_combo(s)

    def _register_landing_combo(self, s: GameState, landed_level: int) -> None:
        if s.jump_takeoff_level is None:
            return
        levels_gained = landed_level - s.jump_takeoff_level
        if levels_gained >= COMBO_MIN_LEVELS_GAIN:
            if s.combo_chain > 0:
                s.combo_chain += 1
            else:
                s.combo_chain = 1
            s.combo_timer = COMBO_TIMER_DURATION
        elif s.combo_chain > 0:
            self._end_combo(s)

    def _end_combo(self, s: GameState) -> None:
        if s.combo_chain <= 0:
            s.combo_timer = 0.0
            return
        points = COMBO_BASE_SCORE * s.combo_chain * s.combo_chain
        s.score += points
        s.last_combo_score = points
        s.combo_chain = 0
        s.combo_timer = 0.0

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
        s.wall_chain = 0
        s.highest_level = max(s.highest_level, plat.level)
        self._register_landing_combo(s, plat.level)
        s.jump_takeoff_level = plat.level

    def _trigger_fall(self, s: GameState, reason: str) -> None:
        self._end_combo(s)
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
