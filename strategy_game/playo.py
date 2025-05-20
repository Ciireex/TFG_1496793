import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPPOA2C import StrategyEnvPPOA2C
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
blue_model = PPO.load("models/ppopruebablue.zip")
red_model = PPO.load("models/ppopruebared.zip")

# Crear entorno base
env = StrategyEnvPPOA2C()

# Inicializar renderer
renderer = Renderer(board_size=env.board_size)

# Iniciar partida
obs, _ = env.reset()
done = False

while not done:
    current_team = env.current_player

    # Elegir acción del modelo correspondiente
    if current_team == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    # Aplicar acción
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Unidad activa
    team_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    active_unit = team_units[env.active_unit_index] if env.active_unit_index < len(team_units) else None

    # Obstáculos
    blocked = [(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]]

    # Dibujar estado actual
    renderer.draw_board(
        units=env.units,
        blocked_positions=blocked,
        capture_point=env.capture_point,
        capture_progress=env.capture_progress,
        active_unit=active_unit,
        terrain=None  # Si tu entorno tiene terreno especial, reemplaza con env.terrain
    )

    print(f"Equipo {'AZUL' if current_team == 0 else 'ROJO'} - Acción: {action} - Recompensa: {reward:.2f}")
    time.sleep(0.4)

print("🎯 Partida finalizada.")
