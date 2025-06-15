import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
import numpy as np
import torch
from stable_baselines3 import A2C
import gymnasium as gym

from gym_strategy.envs.StrategyEnv_TransferMedium_1v1_Archers import StrategyEnv_TransferMedium_1v1_Archers
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.core.Renderer import Renderer  # Asegúrate de que la ruta es correcta

# === Wrapper para que ROJO sea tonto ===
class DummyRedWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 0:  # Azul juega normalmente
            return self.env.step(action)
        else:  # Rojo hace siempre acción 0 (no moverse)
            return self.env.step(0)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# === MAIN: reproducir partida ===
if __name__ == "__main__":
    model_path_blue = "./logs/a2c_blue_vs_dummy_red_fase1/final_model"

    # Crear entorno con rojo dummy
    raw_env = StrategyEnv_TransferMedium_1v1_Archers()
    env = DummyRedWrapper(raw_env)
    renderer = Renderer(board_size=raw_env.board_size)

    # Cargar modelo A2C entrenado del azul
    model_blue = A2C.load(model_path_blue, custom_objects={"features_extractor_class": CustomCNN})
    model_blue.policy.eval()

    obs, _ = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        if env.env.current_player == 0:
            obs_tensor = torch.tensor(obs).unsqueeze(0)
            action, _ = model_blue.predict(obs_tensor, deterministic=True)
        else:
            action = 0  # Dummy red

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
