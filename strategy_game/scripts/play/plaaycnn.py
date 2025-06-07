import sys
import os
import pygame
import numpy as np

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
model_blue = PPO.load("models/ppo_castle_blue", device="cpu")
model_red = PPO.load("models/ppo_castle_red", device="cpu")

# Crear entorno y renderizador
env = StrategyEnv_Castle(use_obstacles=True)
renderer = Renderer(width=720, height=560, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "izquierda",
    2: "derecha",
    3: "arriba",
    4: "abajo"
}

print("\n🎮 Visualizando PPO Azul vs PPO Rojo (castillo)")
print("⏩ Pulsa ESPACIO para avanzar cada acción. Cierra la ventana para salir.\n")

while not done:
    pygame.event.pump()

    # Obtener unidad activa
    team_units = [u for u in env.units if u.team == env.current_player and u.is_alive()]
    index = env.unit_index_per_team[env.current_player]
    active_unit = team_units[index] if index < len(team_units) else None

    # Dibujar tablero
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=active_unit,
        castle_zone=env.castle_area,
        castle_hp=env.castle_control
    )

    # Esperar tecla
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    # Decidir acción
    if env.current_player == 0:
        action, _ = model_blue.predict(obs, deterministic=True)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🤖 PPO Azul ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")
    else:
        action, _ = model_red.predict(obs, deterministic=True)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🤖 PPO Rojo ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")

    obs, _, done, _, _ = env.step(int(action))

pygame.quit()
print("\n✅ Partida finalizada.")
