import os, sys
import numpy as np
import pygame
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Fase2 import StrategyEnv_Fase2
from gym_strategy.utils.CustomCNN_Pro import CustomCNN
from gym_strategy.core.Renderer import Renderer

from stable_baselines3 import PPO

# === CARGA MODELOS ===
blue_model_path = "./logs/fase2_blue_vs_dummy_red/final_model"
red_model_path = "./logs/fase2_red_vs_dummy_blue/final_model"

blue_model = PPO.load(blue_model_path, custom_objects={"features_extractor_class": CustomCNN})
red_model = PPO.load(red_model_path, custom_objects={"features_extractor_class": CustomCNN})

# === INICIAR ENTORNO ===
env = StrategyEnv_Fase2(use_obstacles=False)  # Cambia a True si quieres ver obstáculos
obs, _ = env.reset()

renderer = Renderer(board_size=env.board_size)

done = False
clock = pygame.time.Clock()
paused = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
            elif event.key == pygame.K_SPACE:
                paused = not paused

    if paused:
        clock.tick(10)
        continue

    current_player = env.current_player
    model = blue_model if current_player == 0 else red_model

    obs_tensor = np.expand_dims(obs, axis=0)
    action, _ = model.predict(obs_tensor, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))

    done = terminated or truncated

    renderer.draw_board(
        units=env.units,
        blocked_positions=env.obstacles,
        active_unit=env._get_active_unit()
    )

    clock.tick(3)  # Control de FPS (puedes subir o bajar esto)

pygame.quit()
