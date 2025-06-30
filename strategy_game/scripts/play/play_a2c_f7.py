import os
import sys
import pygame
import numpy as np
from stable_baselines3 import A2C

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CARGA DE MODELOS ===
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_f7_v2")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_red_f7_v3")

blue_model = A2C.load(BLUE_MODEL_PATH, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
red_model = A2C.load(RED_MODEL_PATH, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")

# === ENTORNO Y RENDER ===
env = Env_Fase7_Terreno()
renderer = Renderer(width=1000, height=600, board_size=env.board_size)

obs, _ = env.reset()
done = False
clock = pygame.time.Clock()

while not done:
    active_unit = env._get_active_unit()
    team = env.current_player

    if team == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    obs, _, done, _, _ = env.step(action)

    # === Pintar ===
    blocked = (env.terrain == 99).astype(np.int8)
    renderer.draw_board(units=env.units,
                        terrain=env.terrain,
                        blocked_positions=blocked,
                        active_unit=active_unit)

    clock.tick(8)  

pygame.quit()
