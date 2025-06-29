import os
import sys
import pygame
import numpy as np
from stable_baselines3 import A2C

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase2_Soldiers6x4
from gym_strategy.core.Renderer import Renderer

# === RUTAS DE MODELOS ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_f2_v1.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_red_f2_v1.zip")

# === CARGA DE MODELOS ===
blue_model = A2C.load(BLUE_MODEL_PATH, device="auto")
red_model = A2C.load(RED_MODEL_PATH, device="auto")

# === ENTORNO Y RENDER ===
env = Env_Fase2_Soldiers6x4()
renderer = Renderer(width=700, height=500, board_size=env.board_size)

# === LOOP DE PARTIDA ===
obs, _ = env.reset()
done = False

while not done:
    if env.current_player == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    obs, reward, done, _, _ = env.step(action)

    # Convertir posiciones bloqueadas a array 2D para el renderer
    blocked = np.zeros(env.board_size, dtype=np.uint8)
    for x in range(env.board_size[0]):
        for y in range(env.board_size[1]):
            if env.terrain[x, y] == 99:
                blocked[x, y] = 1

    renderer.draw_board(
        units=env.units,
        blocked_positions=blocked,
        active_unit=env._get_active_unit(),
        highlight_attack=True,
        terrain=env.terrain
    )
    pygame.time.delay(150)

pygame.quit()
