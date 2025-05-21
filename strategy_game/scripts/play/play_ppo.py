import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.core.Renderer import Renderer

# Cargar modelos PPO entrenados
model_blue = PPO.load("models/ppoblue_v2")
model_red = PPO.load("models/ppored_v2")

# Inicializar entorno y renderer
env = StrategyEnv(use_obstacles=True)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Diccionario de acciones para mostrar
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
    if current_team == 0:
        action, _ = model_blue.predict(obs, deterministic=True)
        agent_name = "PPO v2 (azul)"
    else:
        action, _ = model_red.predict(obs, deterministic=True)
        agent_name = "PPO v2 (rojo)"

    action = int(action)
    action_text = f"{agent_name}: "
    if phase == "move":
        action_text += f"MUEVE {action_names.get(action, '?')}"
    elif phase == "attack":
        action_text += f"ATACA {action_names.get(action, '?')}"
    print(action_text)

    obs, reward, done, _, _ = env.step(action)

    # Dibujar el tablero
    renderer.draw_board(
        units=env.units,
        blocked_positions={
            (x, y) for x in range(env.board_size[0])
            for y in range(env.board_size[1]) if env.obstacles[x, y]
        },
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
  