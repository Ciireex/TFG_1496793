import time
import pygame
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_Castle_Lite import StrategyEnv_Castle_Lite
from gym_strategy.utils.CustomCNN_Pro import CustomCNN
from gym_strategy.core.Renderer import Renderer  # Asegúrate de que esté en la ruta correcta

# Rutas de los modelos
MODEL_PATH_0 = "./models/ppo_team0"
MODEL_PATH_1 = "./models/ppo_team1"
DELAY = 0.3  # segundos entre turnos

# Cargar entorno y modelos
env = StrategyEnv_Castle_Lite()
model_team0 = PPO.load(MODEL_PATH_0, custom_objects={"features_extractor_class": CustomCNN})
model_team1 = PPO.load(MODEL_PATH_1, custom_objects={"features_extractor_class": CustomCNN})

renderer = Renderer(width=600, height=360, board_size=env.board_size)

obs, _ = env.reset()
done = False
step = 0
MAX_STEPS = 300

while not done and step < MAX_STEPS:
    current_team = env.current_player
    if current_team == 0:
        action, _ = model_team0.predict(obs, deterministic=True)
    else:
        action, _ = model_team1.predict(obs, deterministic=True)

    obs, reward, done, _, _ = env.step(action)

    renderer.draw_board(
        units=env.units,
        blocked_positions=np.argwhere(env.obstacles == 1).tolist(),
        active_unit=env._get_active_unit(),
        castle_area=env.castle_area,
        capture_score=(max(env.castle_control, 0), -min(env.castle_control, 0)),
        castle_hp=env.castle_control
    )

    time.sleep(DELAY)
    step += 1

# Esperar cierre de ventana
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()
