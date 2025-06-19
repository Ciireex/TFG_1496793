import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Añadir ruta base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_3v3Soldiers6x4_Obs
from gym_strategy.core.Renderer import Renderer

# Entorno con dos políticas
class DualPolicyEnv(StrategyEnv_3v3Soldiers6x4_Obs):
    def __init__(self, model_blue, model_red):
        super().__init__()
        self.model_blue = model_blue
        self.model_red = model_red

    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:
            action, _ = self.model_blue.predict(obs, deterministic=True)
        else:
            action, _ = self.model_red.predict(obs, deterministic=True)
        return super().step(action)

def main():
    print("🧠 Cargando modelos entrenados...")
    model_blue = PPO.load("../models/ppo_blue_vs_heuristicred/ppo_blue_vf2.zip")
    model_red = PPO.load("../models/ppo_red_vs_frozenblue/ppo_red_vf2.zip")
    print("✅ Modelos cargados correctamente.")

    print("🎮 Inicializando entorno...")
    env = DualPolicyEnv(model_blue=model_blue, model_red=model_red)
    renderer = Renderer(width=600, height=400, board_size=env.board_size)

    seed = int(time.time() * 1000) % 2**32
    obs, _ = env.reset(seed=seed)
    print(f"🌱 Semilla de partida: {seed}")

    done = False
    clock = pygame.time.Clock()

    while not done:
        obs = env._get_obs()
        obs, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=env.terrain == 99,
            active_unit=env._get_active_unit()
        )

        clock.tick(4)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
