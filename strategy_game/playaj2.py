import time
import pygame
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gym_strategy.envs.StrategyEnvPPOA2C import StrategyEnvPPOA2C
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
model_blue = DQN.load("models/dqn_blue_dqn_ppov10.zip")
model_red = DQN.load("models/dqn_red_dqn_ppov10.zip")

# Inicializar entorno original (sin wrapper flatten)
env = StrategyEnvPPOA2C()
renderer = Renderer(width=700, height=500, board_size=env.board_size)

# Función para renderizar estado del entorno
def render_env(env):
    obstacles = {(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]}
    active_unit = env.units[env.active_unit_index] if env.active_unit_index < len(env.units) else None
    renderer.draw_board(
        units=env.units,
        blocked_positions=obstacles,
        capture_point=env.capture_point,
        capture_progress=env.capture_progress,
        capture_max=env.capture_turns_required,
        active_unit=active_unit,
        highlight_attack=True
    )

# Inicializar entorno para observación aplanada para el modelo
flat_env = FlattenObservation(env)
obs, _ = flat_env.reset()
done = False

# Loop de juego
while not done:
    current_team = env.current_player
    model = model_blue if current_team == 0 else model_red
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = flat_env.step(action)
    done = terminated or truncated

    render_env(env)

    unit = env.units[env.active_unit_index] if env.active_unit_index < len(env.units) else None
    if unit:
        print(f"Turno {'AZUL' if current_team == 0 else 'ROJO'} - {unit.unit_type} en {unit.position} - Acción: {action}")

    time.sleep(0.3)

pygame.quit()
