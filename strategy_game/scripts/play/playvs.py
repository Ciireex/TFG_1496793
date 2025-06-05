import sys
import os
import pygame
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import DQN, PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.core.Renderer import Renderer

# Cargar modelos
blue_model = DQN.load("models/dqn_blue_v4_cycle10.zip", device="cpu")
red_model = PPO.load("models/pporojo_vs_ppoblue_v3_ciclo10.zip", device="cpu")

# Inicializar entorno y renderer
env = StrategyEnv(use_obstacles=True)
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

    # Elegir acción
    if team == 0:
        obs_flat = obs.flatten().reshape(1, -1)
        action, _ = blue_model.predict(obs_flat, deterministic=True)
        agent_name = "DQN Azul v4"
    else:
        action, _ = red_model.predict(obs, deterministic=True)
        agent_name = "PPO Rojo v3"

    # ✅ Parche: asegurar que action sea int puro
    action = int(action) if isinstance(action, (np.ndarray, list)) else action

    print(f"{agent_name}: {phase.upper()} → {action_names.get(action, '?')}")
    obs, _, done, _, _ = env.step(action)

    # Dibujar tablero
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
