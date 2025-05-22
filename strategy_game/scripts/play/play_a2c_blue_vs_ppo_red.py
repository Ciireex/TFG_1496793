import sys
import os
import time
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.core.Renderer import Renderer

# === Cargar modelos ===
a2c_blue = A2C.load("models/a2cblue_vs_ppored_v1", device="cpu")
ppo_red = PPO.load("models/pporojo_vs_heuristic_curriculum_v1", device="cpu")

# === Crear entorno y renderer ===
env = StrategyEnv(use_obstacles=True)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

# === Loop de juego ===
obs, info = env.reset()
renderer.draw_board(env.board.units)
time.sleep(1)

done = False
while not done:
    if env.current_player == 0:
        action, _ = a2c_blue.predict(obs, deterministic=True)
    else:
        action, _ = ppo_red.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    renderer.draw_board(env.board.units)
    time.sleep(0.3)

    done = terminated or truncated

print("✅ Partida terminada.")
pygame.time.wait(2000)
pygame.quit()
