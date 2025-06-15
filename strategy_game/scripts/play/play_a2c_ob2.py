import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import pygame
import numpy as np
import torch as th
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnv_Fase3 import StrategyEnv_Fase3
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.core.Renderer import Renderer

# === CARGAR MODELOS ===
model_blue = A2C.load(
    "./logs/a2c_blue_fase4_obstacles/final_model",
    custom_objects={"features_extractor_class": CustomCNN}
)
model_red = A2C.load(
    "./logs/a2c_red_fase4_obstacles/final_model",
    custom_objects={"features_extractor_class": CustomCNN}
)

model_blue.policy.eval()
model_red.policy.eval()

# === INICIAR ENTORNO CON OBSTÁCULOS ===
env = StrategyEnv_Fase3(obstacle_count=10)
obs, _ = env.reset()

# === INICIAR RENDERER ===
renderer = Renderer(width=800, height=600, board_size=env.board_size)

# === BUCLE DE JUEGO ===
terminated = False
while not terminated:
    renderer.draw_board(
        units=env.units,
        blocked_positions=env.obstacles,
        active_unit=env._get_active_unit(),
        terrain=None,
        castle_area=getattr(env, 'castle_area', None),
        capture_score=getattr(env, 'capture_score', None),
        castle_hp=getattr(env, 'castle_hp', None)
    )

    pygame.time.wait(300)

    obs_tensor = th.tensor(obs).unsqueeze(0)
    if env.current_player == 0:
        action, _ = model_blue.predict(obs_tensor, deterministic=True)
    else:
        action, _ = model_red.predict(obs_tensor, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(int(action))

pygame.quit()
