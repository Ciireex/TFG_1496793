import os
import sys
import time
import torch
import pygame
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4
from gym_strategy.core.Renderer import Renderer

def main():
    print("🧠 Cargando modelo PPO...")
    model = PPO.load("./models/ppo_blue_vs_randomred/ppo_blue_f0_v0")
    print("✅ Modelo PPO cargado correctamente.")

    print("🎮 Inicializando entorno...")
    env = StrategyEnv_2v2Soldiers4x4()

    renderer = Renderer(width=500, height=500, board_size=env.board_size)

    # Usar una semilla variable (por tiempo actual) para partidas distintas
    seed = int(time.time()) % 10000
    obs, _ = env.reset(seed=seed)

    done = False
    clock = pygame.time.Clock()

    while not done:
        if env.current_player == 0:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Comportamiento aleatorio válido para el rojo
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            active_unit=env._get_active_unit()
        )

        clock.tick(4)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
