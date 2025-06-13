import os, sys, time, torch
import gymnasium as gym
import pygame

# Ajustar path si hace falta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1_Archers import StrategyEnv_TransferSmall_1v1_Archers
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.core.Renderer import Renderer

# === Cargar modelos ===
blue_model = PPO.load("./logs/transfer_blue_vs_fixed_red/final_model", custom_objects={"features_extractor_class": CustomCNN})
red_model = PPO.load("./logs/transfer_red_vs_fixed_blue/final_model", custom_objects={"features_extractor_class": CustomCNN})

# === Inicializar entorno ===
env = StrategyEnv_TransferSmall_1v1_Archers()
renderer = Renderer(width=600, height=400, board_size=env.board_size)

obs, info = env.reset()
terminated = False
truncated = False

# === Bucle principal ===
while not (terminated or truncated):
    active_unit = env._get_active_unit()
    if env.current_player == 0:
        action, _ = blue_model.predict(torch.tensor(obs).unsqueeze(0), deterministic=True)
    else:
        action, _ = red_model.predict(torch.tensor(obs).unsqueeze(0), deterministic=True)

    obs, reward, terminated, truncated, info = env.step(int(action))

    # Dibujar el estado actual
    renderer.draw_board(
        units=env.units,
        active_unit=active_unit,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    time.sleep(0.3)

# Esperar antes de cerrar
time.sleep(2)
pygame.quit()
