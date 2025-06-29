import os
import sys
import pygame
from stable_baselines3 import DQN

# Rutas base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase3_Obstaculos
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.core.Renderer import Renderer

# === CONFIGURACIÓN ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
MODEL_BLUE_PATH = os.path.join(MODEL_DIR, "dqn_blue_f3_v2")
MODEL_RED_PATH = os.path.join(MODEL_DIR, "dqn_red_f3")

# === CARGA DE MODELOS ===
print("🧠 Cargando modelos DQN F3...")
model_blue = DQN.load(MODEL_BLUE_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
model_red = DQN.load(MODEL_RED_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
print("✅ Modelos DQN cargados correctamente.")

# === ENTORNO CON AMBOS MODELOS ===
class DualPolicyEnv(Env_Fase3_Obstaculos):
    def __init__(self):
        super().__init__()

    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:  # azul
            action, _ = model_blue.predict(obs, deterministic=True)
        else:  # rojo
            action, _ = model_red.predict(obs, deterministic=True)
        return super().step(action)

# === MAIN LOOP ===
def main():
    pygame.init()
    env = DualPolicyEnv()
    obs, _ = env.reset()
    renderer = Renderer(width=700, height=500, board_size=env.board_size)

    done = False
    clock = pygame.time.Clock()

    while not done:
        _, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=(env.terrain == 99).astype(int),  # Obstáculos como máscara
            active_unit=env._get_active_unit()
        )
        clock.tick(2)

    pygame.quit()
    print("🎮 Partida finalizada.")

if __name__ == "__main__":
    main()
