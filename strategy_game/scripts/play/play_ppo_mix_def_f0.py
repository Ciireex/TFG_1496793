import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Añadir la ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase2_Soldiers6x4
from gym_strategy.core.Renderer import Renderer

# === ENV CON POLÍTICAS POR BANDO ===
class DualPolicyEnv(Env_Fase2_Soldiers6x4):
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
    BLUE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_blue_def_f1.zip"))
    RED_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_red_def_f1.zip"))

    print("🧠 Cargando modelos entrenados...")
    model_blue = PPO.load(BLUE_PATH)
    model_red = PPO.load(RED_PATH)
    print("✅ Modelos cargados correctamente.")

    print("🎮 Inicializando entorno Fase 2 (6x4 soldados)...")
    env = DualPolicyEnv(model_blue=model_blue, model_red=model_red)
    renderer = Renderer(width=480, height=320, board_size=env.board_size)

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
