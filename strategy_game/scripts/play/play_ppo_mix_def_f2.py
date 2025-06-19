import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase3_Obstaculos
from gym_strategy.core.Renderer import Renderer

# === Entorno donde cada modelo controla su propio equipo ===
class DualPolicyEnv(Env_Fase3_Obstaculos):
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
    MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))

    # Cargar modelos
    print("🧠 Cargando modelos PPO entrenados...")
    model_blue = PPO.load(os.path.join(MODEL_DIR, "ppo_blue_def_f2"))
    model_red = PPO.load(os.path.join(MODEL_DIR, "ppo_red_def_f2"))
    print("✅ Modelos BLUE y RED cargados.")

    # Inicializar entorno y renderer
    env = DualPolicyEnv(model_blue=model_blue, model_red=model_red)
    renderer = Renderer(width=480, height=384, board_size=env.board_size)

    # Reset y render inicial
    seed = int(time.time() * 1000) % 2**32
    obs, _ = env.reset(seed=seed)
    print(f"🌱 Semilla de partida: {seed}")

    clock = pygame.time.Clock()
    done = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=(env.terrain == 99),
            terrain=env.terrain,
            active_unit=env._get_active_unit()
        )

        clock.tick(4)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
