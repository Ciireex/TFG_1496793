import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Rutas del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Def import Env_Fase5_Knight
from gym_strategy.core.Renderer import Renderer

# === MODELOS ===
BLUE_MODEL = "ppo_blue_def_f4"
RED_MODEL = "ppo_red_def_f4"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

# === Cargar modelos entrenados ===
print("🔵 Cargando modelo azul:", BLUE_MODEL)
model_blue = PPO.load(os.path.join(MODEL_DIR, BLUE_MODEL), device="cpu")
print("🔴 Cargando modelo rojo:", RED_MODEL)
model_red = PPO.load(os.path.join(MODEL_DIR, RED_MODEL), device="cpu")

# === Crear entorno y renderer ===
env = Env_Fase5_Knight()
renderer = Renderer(width=700, height=400, board_size=env.board_size)

obs = env.reset()[0]
done = False

# === Loop de juego ===
while not done:
    renderer.draw_board(
        units=env.units,
        blocked_positions=(env.terrain == 99),
        active_unit=env._get_active_unit(),
        terrain=env.terrain
    )
    pygame.display.flip()
    time.sleep(0.3)

    # Elegir acción según el jugador
    if env.current_player == 0:
        action, _ = model_blue.predict(obs, deterministic=True)
    else:
        action, _ = model_red.predict(obs, deterministic=True)

    obs, reward, done, _, info = env.step(action)

# === Mostrar estado final ===
renderer.draw_board(
    units=env.units,
    blocked_positions=(env.terrain == 99),
    active_unit=None,
    terrain=env.terrain
)
pygame.display.flip()
print("🎯 Partida finalizada.")
time.sleep(3)
pygame.quit()
