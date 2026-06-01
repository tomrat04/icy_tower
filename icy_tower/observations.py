from __future__ import annotations

import numpy as np

from icy_tower.config import (
    LEVEL_HEIGHT,
    OBS_PLATFORMS_ABOVE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WIN_LEVEL,
)
from icy_tower.game import GameState

BASE_OBS_DIM = 6
PLATFORM_FEAT_DIM = 4
OBS_DIM = BASE_OBS_DIM + OBS_PLATFORMS_ABOVE * PLATFORM_FEAT_DIM


def build_observation(state: GameState) -> np.ndarray:
    """Wektor stanu: gracz + 10 platform (poziomy) nad ostatnią platformą odbicia."""
    p = state.player
    cam = state.camera_y
    ref_y = state.reference_platform_y if state.reference_platform_y is not None else p.feet_y
    ref_level = (
        state.reference_platform_level
        if state.reference_platform_level is not None
        else 0
    )

    platform_feats: list[float] = []
    for offset in range(1, OBS_PLATFORMS_ABOVE + 1):
        target_level = ref_level + offset
        level_plats = [pl for pl in state.world.platforms if pl.level == target_level]
        if level_plats:
            pl = min(
                level_plats,
                key=lambda plat: abs(plat.x + plat.width / 2 - p.center_x),
            )
            platform_feats.extend(
                [
                    pl.x / SCREEN_WIDTH,
                    (pl.y - cam) / SCREEN_HEIGHT,
                    pl.width / SCREEN_WIDTH,
                    pl.level / float(WIN_LEVEL),
                ]
            )
        else:
            platform_feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(
        [
            p.x / SCREEN_WIDTH,
            (p.y - cam) / SCREEN_HEIGHT,
            p.vx / 500.0,
            p.vy / 800.0,
            (ref_y - p.feet_y) / LEVEL_HEIGHT,
            state.highest_level / float(WIN_LEVEL),
            *platform_feats,
        ],
        dtype=np.float32,
    )
