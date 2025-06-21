import os
import sys
import time
import pygame
from stable_baselines3 import A2C

# Añadir ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase3_Archer6x4
from gym_strategy.core.Renderer import Renderer

# Entorno con políticas por equipo
class DualPolicyEnv(Env_Fase3_Archer6x4):
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

    # === Cargar modelos ===
    BLUE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/a2c_blue_def_f1_retrain2.zip"))
    RED_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/a2c_red_def_f1_retrain.zip"))

    print("🧠 Cargando modelos A2C...")
    model_blue = A2C.load(BLUE_PATH)
    model_red = A2C.load(RED_PATH)
    print("✅ Modelos cargados correctamente.")

    print("🎯 Inicializando entorno Fase 3 (6x4 con arqueros y obstáculos)...")
    env = DualPolicyEnv(model_blue=model_blue, model_red=model_red)
    renderer = Renderer(width=480, height=320, board_size=env.board_size)

    # 🪄 Generar semilla aleatoria cada ejecución
    seed = int(time.time() * 1000) % 2**32
    obs, _ = env.reset(seed=seed)
    print(f"🌱 Semilla aleatoria usada: {seed}")

    done = False
    clock = pygame.time.Clock()

    while not done:
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
