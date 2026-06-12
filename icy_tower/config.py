"""Stałe gry Icy Tower."""

import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

PLAYER_WIDTH = 28
PLAYER_HEIGHT = 36

LEVEL_HEIGHT = 110
PLATFORM_HEIGHT = 14

GRAVITY = 1200.0
MAX_FALL_SPEED = 900.0

# Ruch poziomy — stała prędkość (bez „boostów” na boki)
MOVE_SPEED = 350.0

# Rozbieg — trzymasz ←/→ na ziemi → rośnie pęd PIONOWY na następny skok
RUN_MOMENTUM_BUILD = 1.8
RUN_MOMENTUM_DECAY = 1.1
RUN_MOMENTUM_FOR_MAX_JUMP = 1.0
# Mnożnik vy przy pełnym pędzie: vy = STAND * (1 + MOMENTUM_JUMP_BOOST * momentum)
MOMENTUM_JUMP_BOOST = 0.35
JUMP_VELOCITY_STAND = -math.sqrt(2 * GRAVITY * LEVEL_HEIGHT) * 1.25
JUMP_VELOCITY_MAX = -math.sqrt(2 * GRAVITY * 2 * LEVEL_HEIGHT)


def _jump_air_time(vy0: float, levels_up: int) -> float:
    """Czas lotu do lądowania `levels_up` pięter wyżej (przy danym vy0)."""
    target = levels_up * LEVEL_HEIGHT
    a = 0.5 * GRAVITY
    disc = vy0 * vy0 - 4 * a * target
    if disc <= 0:
        return 0.85
    t1 = (-vy0 - math.sqrt(disc)) / (2 * a)
    t2 = (-vy0 + math.sqrt(disc)) / (2 * a)
    return max(t1, t2)


# Maks. przesunięcie X (stopy) podczas skoku — do walidacji mapy
JUMP_REACH_ONE_LEVEL = MOVE_SPEED * _jump_air_time(JUMP_VELOCITY_STAND, 1) + PLAYER_WIDTH + 16
JUMP_REACH_TWO_LEVELS = MOVE_SPEED * _jump_air_time(JUMP_VELOCITY_MAX, 2) + PLAYER_WIDTH + 16

# Wall jump — jednorazowy boost vy; po 2. odbiciu lub lądowaniu reset
WALL_JUMP_MARGIN = 22.0
WALL_JUMP_MIN_MOMENTUM = 0.1
WALL_VY_BOOST = 1.15
MAX_WALL_BOUNCES_PER_AIR = 1

# Tarcie poziome na platformie (puszczenie klawiszy)
GROUND_DECEL = 2600.0
GROUND_STOP_SPEED = 25.0

# Kamera — płynne podążanie + powolny auto-scroll (presja)
CAMERA_PLAYER_RATIO = 0.58
CAMERA_SMOOTH = 6.5
CAMERA_SMOOTH_AIR = 9.0
AUTO_SCROLL_SPEED = 38.0
SCROLL_DEATH_MARGIN = 40.0

# Mapa — 100 poziomów easy (0–99), 100 hard (100–199)
EASY_LEVELS = 100
HARD_MODE_LEVEL = 100
EARLY_PLATFORM_WIDTHS = (190, 210, 230, 250)
HARD_PLATFORM_WIDTHS = (65, 75, 85, 95)
HARD_TWO_PLATFORM_CHANCE = 0.25

WIN_LEVEL = 200
# Trening: losowy start 0..TRAIN_START_LEVEL_MAX (~43% epizodów na hard, poz. 100+)
TRAIN_START_LEVEL_MAX = 175
OBS_PLATFORMS_NEAR = 12
ROWS_AHEAD = 25
ROWS_BEHIND = 0
MAP_SIDE_MARGIN = 24.0

FPS = 60
DT = 1.0 / FPS

COLOR_BG = (25, 28, 45)
COLOR_PLATFORM = (90, 200, 120)
COLOR_PLAYER = (255, 220, 80)
COLOR_TEXT = (230, 230, 240)
COLOR_DEATH = (220, 70, 70)
COLOR_WIN = (80, 220, 160)
COLOR_BAR_BG = (60, 55, 80)
COLOR_RUN_BAR = (100, 200, 255)
