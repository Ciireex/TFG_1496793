import sys
import os
import time
import pygame
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.core.Renderer import Renderer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Cargar modelos
model_blue = DQN.load("models/dqn_blue_vs_heuristic_v1", device="cpu")
model_red = DQN.load("models/dqn_red_vs_heuristic_v1", device="cpu")

# Crear entorno
env = StrategyEnvPPOA2C2()
env = FlattenObservation(env)
renderer = Renderer(width=700, height=500, board_size=env.env.board_size)

obs, _ = env.reset()
done = False

print("🎮 DQN Azul vs DQN Rojo")

while not done:
    time.sleep(0.4)

    current_team = env.env.current_player
    idx = env.env.unit_index_per_team[current_team]
    my_units = [u for u in env.env.units if u.team == current_team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    # Elegir acción del modelo correspondiente
    if current_team == 0:
        action, _ = model_blue.predict(obs, deterministic=True)
        name = "DQN (azul)"
    else:
        action, _ = model_red.predict(obs, deterministic=True)
        name = "DQN (rojo)"

    phase = env.env.phase
    dir_names = {0: "pasar", 1: "←", 2: "→", 3: "↑", 4: "↓"}
    action_str = dir_names.get(int(action), "?")
    print(f"{name} - {phase.upper()} → {action_str}")

    obs, reward, done, _, _ = env.step(action)

    # Dibujar
    blocked = [(x, y) for x in range(env.env.board_size[0])
                      for y in range(env.env.board_size[1]) if env.env.obstacles[x, y]]
    renderer.draw_board(
        units=env.env.units,
        blocked_positions=blocked,
        active_unit=unit,
        capture_point=env.env.capture_point,
        capture_score=env.env.capture_progress,
        max_capture=env.env.capture_turns_required
    )

print("✅ Partida finalizada")
pygame.quit()
