import time
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from icy_tower.config import DT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from icy_tower.env import IcyTowerEnv
from icy_tower.game import GameStatus, IcyTowerGame
from icy_tower.observations import build_observation
from icy_tower.render import GameRenderer
from icy_tower.wrappers import IcyTowerVecWrapper

def run_test(model_path="models/best/best_model.zip", fps=FPS, pause_on_end=2.0):
    vec_path = "models/best/vecnormalize.pkl"

    dummy = DummyVecEnv([lambda: IcyTowerVecWrapper(IcyTowerEnv(seed=0))])
    vecnorm = VecNormalize.load(vec_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = PPO.load(model_path)
    game = IcyTowerGame(seed=0)
    state = game.reset(seed=0)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Icy Tower — test")
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen)

    running = True
    pause_until = 0.0
    episode = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = game.reset(seed=0)
                    pause_until = 0.0
                    episode += 1

        if time.time() < pause_until:
            renderer.draw(state)
            pygame.display.flip()
            clock.tick(fps)
            continue

        if state.status == GameStatus.PLAYING:
            obs = build_observation(state)
            obs = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            move_left = action in (1, 4)
            move_right = action in (2, 5)
            jump = action in (3, 4, 5)

            state = game.step(DT, move_left, move_right, jump)

        elif state.status == GameStatus.FALLING:
            state = game.step(DT, False, False)
            if state.player.y - state.camera_y > SCREEN_HEIGHT + 120:
                print(f"Epizod {episode}: poziom {state.highest_level}, {state.death_reason}")
                pause_until = time.time() + pause_on_end
                state = game.reset(seed=0)
                episode += 1

        elif state.status == GameStatus.WON:
            print(f"Epizod {episode}: WYGRANA na poziomie {state.highest_level}")
            pause_until = time.time() + pause_on_end
            state = game.reset(seed=0)
            episode += 1

        renderer.draw(state)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

run_test()