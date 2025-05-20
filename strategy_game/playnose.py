import time
import pygame
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPPOA2C import StrategyEnvPPOA2C
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
model_blue = PPO.load("models/ppo_blue_ppov10.zip")
model_red = PPO.load("models/ppo_red_ppov10.zip")

# Inicializar entorno y renderer
env = StrategyEnvPPOA2C()
renderer = Renderer(width=700, height=500, board_size=env.board_size)

# Función para renderizar el entorno completo
def render_env(renderer, env):
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

# Iniciar partida
obs, _ = env.reset()
done = False

while not done:
    current_team = env.current_player
    unit = env.units[env.active_unit_index] if env.active_unit_index < len(env.units) else None

    # Elegir modelo según el equipo
    model = model_blue if current_team == 0 else model_red
    action, _ = model.predict(obs, deterministic=True)

    # Ejecutar acción
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Dibujar entorno actualizado
    render_env(renderer, env)

    # Mostrar info por consola
    if unit:
        print(f"Turno {'AZUL' if current_team == 0 else 'ROJO'} - {unit.unit_type} en {unit.position} - Acción: {action}")

    time.sleep(0.3)

pygame.quit()
