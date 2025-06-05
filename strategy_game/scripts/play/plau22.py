import sys
import os
import pygame
import numpy as np
import time

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_V5 import StrategyEnv_V5
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

env = StrategyEnv_V5(use_obstacles=True)
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Acciones más claras (sin flechas dobles)
action_names = {
    0: "pasar",
    1: "izquierda",
    2: "derecha",
    3: "arriba",
    4: "abajo"
}

print("🎮 Controles: [0] pasar, [1] izquierda, [2] derecha, [3] arriba, [4] abajo")
print("Pulsa una tecla numérica para elegir la acción. Cierra la ventana para salir.")

while not done:
    pygame.event.pump()

    active_unit = next((u for i, u in enumerate(env.units) if u.team == env.current_player and i == env.unit_index_per_team[env.current_player]), None)
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=active_unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

    if env.current_player == 0:
        # Turno del jugador humano
        waiting = True
        action = None
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        action = 0
                    elif event.key == pygame.K_1:
                        action = 1
                    elif event.key == pygame.K_2:
                        action = 2
                    elif event.key == pygame.K_3:
                        action = 3
                    elif event.key == pygame.K_4:
                        action = 4
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
                    if action is not None:
                        waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"👤 Tú ({env.phase.upper()} | {unit_type}) → {action_names[action]}")
        obs, _, done, _, _ = env.step(action)

    else:
        # Turno de la heurística con delay y detalle
        time.sleep(1.0)
        action = heuristic.get_action(obs)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🤖 Heurística ({env.phase.upper()} | {unit_type}) → {action_names[action]}")
        obs, _, done, _, _ = env.step(action)

pygame.quit()
print("\n🎬 Partida finalizada.")
