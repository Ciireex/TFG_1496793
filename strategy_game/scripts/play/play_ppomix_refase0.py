import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import pygame
import time
import numpy as np
import torch
from stable_baselines3 import PPO

from gym_strategy.envs.StrategyEnv_Fase0_v3 import StrategyEnv_Fase0_v2
from gym_strategy.utils.CustomCNN_Pro import CustomCNN_Pro
from gym_strategy.core.Renderer import Renderer

# === Cargar modelos PPO entrenados ===
model_blue = PPO.load("./models/final_model_blue", custom_objects={"features_extractor_class": CustomCNN_Pro})
model_red = PPO.load("./models/final_model_red", custom_objects={"features_extractor_class": CustomCNN_Pro})

# === Crear entorno y renderer ===
env = StrategyEnv_Fase0_v2()
obs, _ = env.reset()
renderer = Renderer(board_size=env.board_size)

done = False
clock = pygame.time.Clock()

# === Bucle de juego ===
while not done:
    renderer.draw_board(
        units=env.units,
        blocked_positions=env.obstacles,
        active_unit=env._get_active_unit()
    )

    if env.current_player == 0:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action, _ = model_blue.predict(obs_tensor, deterministic=True)
    else:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action, _ = model_red.predict(obs_tensor, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)
    clock.tick(3)

# Espera unos segundos antes de cerrar ventana
time.sleep(3)
pygame.quit()
