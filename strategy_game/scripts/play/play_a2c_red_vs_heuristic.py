import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import pygame
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Cargar modelo A2C entrenado (equipo rojo)
model = A2C.load("models/a2cred_vs_heuristic_curriculum_v1")

# Inicializar entorno y heurística
env = StrategyEnv(use_obstacles=True)
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Diccionario de acciones
action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("Presiona ESPACIO para avanzar por cada unidad.")

while not done:
    # Esperar a pulsar espacio
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

    current_team = env.current_player
    phase = env.phase
    idx = env.unit_index_per_team[current_team]
    my_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    # Elegir acción
    if current_team == 1:
        action, _ = model.predict(obs, deterministic=True)
        agent_name = "A2C (rojo)"
    else:
        action = heuristic.get_action(obs)
        agent_name = "Heurística (azul)"

    action = int(action)
    action_text = f"{agent_name}: "
    if phase == "move":
        action_text += f"MUEVE {action_names.get(action, '?')}"
    elif phase == "attack":
        action_text += f"ATACA {action_names.get(action, '?')}"
    print(action_text)

    obs, reward, done, _, _ = env.step(action)

    # Dibujar
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
 