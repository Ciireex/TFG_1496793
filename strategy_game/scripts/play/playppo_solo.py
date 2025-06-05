import sys
import os
import pygame
import numpy as np

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_V4 import StrategyEnv_V4
from gym_strategy.core.Renderer import Renderer

# Cargar el modelo PPO entrenado
model = PPO.load("models/ppo_capture_v1c", device="cpu")

# Crear entorno sin enemigos
env = StrategyEnv_V4(use_obstacles=True, only_blue=True)
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

    # Esperar a que el jugador pulse ESPACIO
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

    # Obtener acción del modelo
    action, _ = model.predict(obs, deterministic=True)
    action = int(action) if isinstance(action, (np.ndarray, list)) else action
    print(f"PPO Azul v1c: {phase.upper()} → {action_names.get(action, '?')}")

    # Aplicar acción
    obs, _, done, _, _ = env.step(action)

    # Renderizar
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
  