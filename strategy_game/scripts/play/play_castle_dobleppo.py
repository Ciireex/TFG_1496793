import os
import sys
import pygame
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_Castle_Lite import StrategyEnv_Castle_Minimal
from gym_strategy.core.Renderer import Renderer

# Cargar los dos modelos entrenados
blue_model = PPO.load("models/ppo_blue_latest.zip", device="cpu")
red_model = PPO.load("models/ppo_red_latest.zip", device="cpu")

# Crear el entorno base
env = StrategyEnv_Castle_Minimal(use_obstacles=True, obstacle_count=4)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False
clock = pygame.time.Clock()

while not done:
    pygame.event.pump()
    current_team = env.current_player

    # Elegir acción del modelo correspondiente
    if current_team == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)

    renderer.draw_board(
        units=env.units,
        blocked_positions=np.argwhere(env.obstacles == 1).tolist(),
        active_unit=[u for u in env.units if u.team == env.current_player and u.is_alive()][env.unit_index_per_team[env.current_player]] if env.unit_index_per_team[env.current_player] < len([u for u in env.units if u.team == env.current_player and u.is_alive()]) else None,
        castle_area=env.castle_area,
        castle_hp=env.castle_control
    )

    clock.tick(1)   # Control de velocidad: 2 FPS para que se vea bien
    done = terminated or truncated

pygame.quit()
print("🏁 Fin de la partida")
