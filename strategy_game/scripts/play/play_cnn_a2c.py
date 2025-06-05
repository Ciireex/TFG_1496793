import sys
import os
import pygame
import numpy as np

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnv_CNN import StrategyEnv_CNN
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Cargar el modelo entrenado A2C
model = A2C.load("models/a2c_cnn_blue", device="cpu")  # Cambia a "cuda" si usaste GPU

# Crear entorno y componentes
env = StrategyEnv_CNN(use_obstacles=True)
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "izquierda",
    2: "derecha",
    3: "arriba",
    4: "abajo"
}

print("\n🎮 Visualizando A2C-CNN (azul) vs Heurística (rojo)")
print("⏩ Pulsa ESPACIO para avanzar cada acción. Cierra la ventana para salir.\n")

while not done:
    pygame.event.pump()

    # Obtener unidad activa
    team_units = [u for u in env.units if u.team == env.current_player and u.is_alive()]
    index = env.unit_index_per_team[env.current_player]
    active_unit = team_units[index] if index < len(team_units) else None

    # Dibujar
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=active_unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
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

    # Acción A2C o heurística
    if env.current_player == 0:
        action, _ = model.predict(obs, deterministic=True)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🤖 A2C-CNN Azul ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")
        obs, _, done, _, _ = env.step(int(action))
    else:
        action = heuristic.get_action(obs)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🧠 Heurística Roja ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")
        obs, _, done, _, _ = env.step(int(action))

pygame.quit()
print("\n✅ Partida finalizada.")
