import sys
import os
import pygame
import numpy as np
import time

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_V5 import StrategyEnv_V5
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Cargar el modelo entrenado
model = PPO.load("models/ppo_vs_heuristic_v10", device="cpu")

# Crear entorno con obstáculos
env = StrategyEnv_V5(use_obstacles=True)
renderer = Renderer(width=700, height=500, board_size=env.board_size)
heuristic = HeuristicPolicy(env)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "izquierda",
    2: "derecha",
    3: "arriba",
    4: "abajo"
}

print("🎮 Simulación PPO (azul) vs Heurística (rojo). Cierra la ventana para salir.")

while not done:
    pygame.event.pump()
    active_unit = next((u for i, u in enumerate(env.units)
                        if u.team == env.current_player and i == env.unit_index_per_team[env.current_player]), None)

    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=active_unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

    time.sleep(0.5)  # para ver bien cada acción

    if env.current_player == 0:
        # PPO juega
        action, _ = model.predict(obs, deterministic=True)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🤖 PPO Azul ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")
        obs, _, done, _, _ = env.step(int(action))
    else:
        # Heurística juega
        action = heuristic.get_action(obs)
        unit_type = active_unit.unit_type if active_unit else "?"
        print(f"🧠 Heurística Roja ({env.phase.upper()} | {unit_type}) → {action_names.get(int(action), '?')}")
        obs, _, done, _, _ = env.step(int(action))

pygame.quit()
print("\n🎬 Partida finalizada.")
