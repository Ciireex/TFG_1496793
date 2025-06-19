import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
import numpy as np
import torch
from stable_baselines3 import A2C
import gymnasium as gym

from gym_strategy.envs.StrategyEnv_TransferMedium_1v1_Archers import StrategyEnv_TransferMedium_1v1_Archers
from gym_strategy.utils.CustomCNN_Pro import CustomCNN
from gym_strategy.core.Renderer import Renderer

# === MAIN ===
if __name__ == "__main__":
    model_path_blue = "./logs/a2c_blue_vs_dummy_red_fase1/final_model"
    model_path_red = "./logs/a2c_red_vs_fixed_blue_fase2/final_model"

    # Crear entorno base
    env = StrategyEnv_TransferMedium_1v1_Archers()
    renderer = Renderer(board_size=env.board_size)

    # Cargar ambos modelos entrenados
    model_blue = A2C.load(model_path_blue, custom_objects={"features_extractor_class": CustomCNN})
    model_blue.policy.eval()

    model_red = A2C.load(model_path_red, custom_objects={"features_extractor_class": CustomCNN})
    model_red.policy.eval()

    obs, _ = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        if env.current_player == 0:
            obs_tensor = torch.tensor(obs).unsqueeze(0)
            action, _ = model_blue.predict(obs_tensor, deterministic=True)
        else:
            obs_tensor = torch.tensor(obs).unsqueeze(0)
            action, _ = model_red.predict(obs_tensor, deterministic=True)

        obs, reward, done, truncated, info = env.step(int(action))

        renderer.draw_board(
            units=env.units,
            blocked_positions=getattr(env, 'blocked_positions', None),
            active_unit=env._get_active_unit(),
            capture_score=getattr(env, 'capture_score', None),
            castle_area=getattr(env, 'castle_area', None),
            castle_hp=getattr(env, 'castle_hp', None)
        )

        clock.tick(3)

    print("🏁 Partida finalizada — Recompensa final:", reward)
