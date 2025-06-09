import os
import sys
import time
import pygame
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.core.Renderer import Renderer

# === Cargar modelos entrenados ===
model_blue = PPO.load("models/ppo_castle_vs_heuristic_continue", device="cpu")  # equipo azul
model_red = PPO.load("models/ppo_red_vs_heuristic_continue", device="cpu")       # equipo rojo

# === Crear entorno y renderer ===
env = StrategyEnv_Castle(use_obstacles=True, obstacle_count=10)
renderer = Renderer(width=700, height=600, board_size=env.board_size)

obs, _ = env.reset()
done = False

while not done:
    # Dibujar estado
    renderer.draw_board(
        units=env.units,
        blocked_positions=list(zip(*env.obstacles.nonzero())),
        active_unit=env.units[env.unit_index_per_team[env.current_player]]
        if env.unit_index_per_team[env.current_player] < len(env.units) else None,
        castle_area=env.castle_area,
        castle_hp=env.castle_control
    )

    time.sleep(0.2)

    # Elegir acción según el equipo activo
    if env.current_player == 0:
        action, _ = model_blue.predict(obs, deterministic=True)
    else:
        action, _ = model_red.predict(obs, deterministic=True)

    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

pygame.quit()
