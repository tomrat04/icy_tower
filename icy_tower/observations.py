from __future__ import annotations

import numpy as np

from icy_tower.config import (
    JUMP_VELOCITY_MAX,
    LEVEL_HEIGHT,
    MAX_FALL_SPEED,
    MAX_WALL_BOUNCES_PER_AIR,
    MOVE_SPEED,
    OBS_PLATFORMS_NEAR,
    RUN_MOMENTUM_FOR_MAX_JUMP,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from icy_tower.game import GameState

# Gracz: pozycja, prędkości, stan gry
OBS_PLAYER_DIM = 15
# Każda platforma: pozycja względem agenta, szerokość, czy można wylądować
OBS_PLATFORM_FEAT_DIM = 7
OBS_DIM = OBS_PLAYER_DIM + OBS_PLATFORMS_NEAR * OBS_PLATFORM_FEAT_DIM


def _player_level(state: GameState) -> int:
    return int(round(-state.player.y / LEVEL_HEIGHT))


def _horizontal_gap_to_platform(px: float, pw: float, plat) -> float:
    """0 = nad platformą; >0 = odległość w bok (znormalizowana później)."""
    if px + pw <= plat.x:
        return plat.x - (px + pw)
    if px >= plat.x + plat.width:
        return px - (plat.x + plat.width)
    return 0.0


def _collect_nearby_platforms(state: GameState) -> list:
    p = state.player
    plats = state.world.get_platforms_near(p.y, radius=LEVEL_HEIGHT * 10)
    plats.sort(key=lambda pl: (abs(pl.y - p.feet_y), abs(pl.x + pl.width / 2 - p.center_x)))
    return plats[:OBS_PLATFORMS_NEAR]


def _platform_features(state: GameState, plat) -> list[float]:
    p = state.player
    cam = state.camera_y
    dx = (plat.x + plat.width / 2 - p.center_x) / SCREEN_WIDTH
    dy_levels = (plat.y - p.feet_y) / LEVEL_HEIGHT
    width = plat.width / SCREEN_WIDTH
    plat_screen_y = (plat.y - cam) / SCREEN_HEIGHT
    gap = _horizontal_gap_to_platform(p.x, p.width, plat) / SCREEN_WIDTH
    can_land = 1.0 if gap == 0.0 and plat.y <= p.feet_y + 6 else 0.0
    return [dx, dy_levels, width, plat_screen_y, gap, can_land, 1.0]


def build_observation(state: GameState) -> np.ndarray:
    p = state.player
    cam = state.camera_y
    screen_x = p.x / SCREEN_WIDTH
    screen_y = (p.y - cam) / SCREEN_HEIGHT
    feet_screen_y = (p.feet_y - cam) / SCREEN_HEIGHT
    world_y = -p.y / (state.win_level * LEVEL_HEIGHT)
    vx = p.vx / MOVE_SPEED
    vy = p.vy / max(abs(JUMP_VELOCITY_MAX), MAX_FALL_SPEED)
    margin_bottom = (SCREEN_HEIGHT - (p.feet_y - cam)) / SCREEN_HEIGHT

    standing_dx = 0.0
    standing_dy = 0.0
    if state.standing_platform_id is not None:
        for plat in state.world.platforms:
            if plat.id == state.standing_platform_id:
                standing_dx = (plat.x + plat.width / 2 - p.center_x) / SCREEN_WIDTH
                standing_dy = (plat.y - p.feet_y) / LEVEL_HEIGHT
                break

    player_feats = [
        screen_x,
        screen_y,
        feet_screen_y,
        world_y,
        vx,
        vy,
        float(state.on_ground),
        state.run_momentum / max(RUN_MOMENTUM_FOR_MAX_JUMP, 1e-6),
        margin_bottom,
        float(state.land_grace > 0.0),
        state.wall_chain / max(MAX_WALL_BOUNCES_PER_AIR, 1),
        max(state.highest_level, state.peak_level) / float(state.win_level),
        standing_dx,
        standing_dy,
        _player_level(state) / float(state.win_level),
    ]

    platform_feats: list[float] = []
    nearby = _collect_nearby_platforms(state)
    for i in range(OBS_PLATFORMS_NEAR):
        if i < len(nearby):
            platform_feats.extend(_platform_features(state, nearby[i]))
        else:
            platform_feats.extend([0.0] * OBS_PLATFORM_FEAT_DIM)

    return np.array([*player_feats, *platform_feats], dtype=np.float32)
