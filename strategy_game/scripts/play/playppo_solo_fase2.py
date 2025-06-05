import sys
import os
import pygame
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_V4 import StrategyEnv_V4
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# === Cargar modelo PPO entrenado ===
model = PPO.load("models/ppo_capture_v1c_v2", device="cpu")

# === Inicializar entorno con enemigo heurístico ===
env = StrategyEnv_V4(use_obstacles=True, only_blue=False, enemy_controller="heuristic")
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("🎮 Pulsa ESPACIO para avanzar cada acción. Cierra la ventana para salir.")

while not done:
    pygame.event.pump()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

    team = env.current_player
    phase = env.phase
    idx = env.unit_index_per_team[team]
    my_units = [u for u in env.units if u.team == team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    if team == 0:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if isinstance(action, (np.ndarray, list)) else action
        print(f"PPO Azul v1c_v2: {phase.upper()} → {action_names.get(action, '?')}")
    else:
        action = heuristic.get_action(obs)
        print(f"Heurística Roja: {phase.upper()} → {action_names.get(action, '?')}")

    obs, _, done, _, _ = env.step(action)

    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("\n🎬 Partida finalizada.")
