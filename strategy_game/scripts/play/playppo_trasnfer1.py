import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
import numpy as np
import torch
from stable_baselines3 import PPO
import gymnasium as gym

from gym_strategy.envs.StrategyEnv_TransferMedium_1v1_Archers import StrategyEnv_TransferMedium_1v1_Archers
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.core.Renderer import Renderer  # Ajusta si lo tienes en otro lado

# === Wrapper para que AZUL sea fijo ===
class FixedBlueWrapper(gym.Wrapper):
    def __init__(self, env, blue_model_path):
        super().__init__(env)
        self.model_blue = PPO.load(blue_model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model_blue.policy.eval()

    def step(self, action):
        if self.env.current_player == 1:  # ROJO aprende
            return self.env.step(action)
        else:  # AZUL fijo
            obs = self.env._get_obs()
            obs_tensor = np.expand_dims(obs, axis=0)
            model_action, _ = self.model_blue.predict(obs_tensor, deterministic=True)
            return self.env.step(int(model_action))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# === MAIN: reproducir partida ===
if __name__ == "__main__":
    # Modelos entrenados
    red_model_path = "./logs/transfer_red_vs_fixed_blue_fase3/final_model"
    blue_model_path = "./logs/transfer_blue_vs_fixed_red_fase2/final_model"

    # Crear entorno
    raw_env = StrategyEnv_TransferMedium_1v1_Archers()
    env = FixedBlueWrapper(raw_env, blue_model_path)
    renderer = Renderer(board_size=raw_env.board_size)

    model_red = PPO.load(red_model_path, custom_objects={"features_extractor_class": CustomCNN})
    model_red.policy.eval()

    obs, _ = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        if env.env.current_player == 1:
            action, _ = model_red.predict(obs, deterministic=True)
        else:
            action = 0  # ignorado, FixedBlueWrapper lo sustituye

        obs, reward, done, truncated, info = env.step(int(action))

        renderer.draw_board(
            units=env.env.units,
            blocked_positions=getattr(env.env, 'blocked_positions', None),
            active_unit=env.env._get_active_unit(),
            capture_score=getattr(env.env, 'capture_score', None),
            castle_area=getattr(env.env, 'castle_area', None),
            castle_hp=getattr(env.env, 'castle_hp', None)
        )

        clock.tick(3)  # 3 FPS

    print("🏁 Partida finalizada — Recompensa final:", reward)
