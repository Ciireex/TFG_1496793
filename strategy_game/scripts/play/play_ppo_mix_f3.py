import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Añadir la ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2Soldiers1Archer_6x4_Obs
from gym_strategy.core.Renderer import Renderer

# Entorno en el que cada modelo actúa en su bando
class DualPolicyEnv(StrategyEnv_2Soldiers1Archer_6x4_Obs):
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
    CURRENT_DIR = os.path.dirname(__file__)
    
    # Cargar modelos
    BLUE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_2soldiers1archer/ppo_2soldiers1archer_final.zip"))
    RED_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_red_vs_frozenblue_2s1a/ppo_red_vs_frozenblue_final.zip"))
    
    print("🧠 Cargando modelos entrenados...")
    model_blue = PPO.load(BLUE_PATH)
    model_red = PPO.load(RED_PATH)
    print("✅ Modelos BLUE y RED cargados.")

    print("🎮 Inicializando entorno 2S1A con obstáculos...")
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
            active_unit=env._get_active_unit(),
            blocked_positions=(env.terrain == 99)
        )

        clock.tick(4)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
