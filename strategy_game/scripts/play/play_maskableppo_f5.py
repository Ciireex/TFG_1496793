import os
import sys
import pygame
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase5_Knight
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.core.Renderer import Renderer

# === CONFIGURACIÓN DE MODELOS ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_blue_f5_v1.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_red_f5_v1.zip")

# === CARGA DE MODELOS ===
print("🧠 Cargando modelos MaskablePPO F5...")
model_blue = MaskablePPO.load(BLUE_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
model_red = MaskablePPO.load(RED_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
print("✅ Modelos cargados correctamente.")

# === ENVOLTORIO PARA ENFRENTAR AZUL VS ROJO ===
class DualMaskableEnvF5(Env_Fase5_Knight):
    def __init__(self):
        super().__init__()

    def step(self, _):
        obs = self._get_obs()
        mask = self.get_action_mask()
        if self.current_player == 0:
            action, _ = model_blue.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _ = model_red.predict(obs, action_masks=mask, deterministic=True)
        return super().step(action)

# === LOOP PRINCIPAL DE VISUALIZACIÓN ===
def main():
    pygame.init()
    env = DualMaskableEnvF5()
    obs, _ = env.reset()
    renderer = Renderer(width=600, height=400, board_size=env.board_size)

    done = False
    clock = pygame.time.Clock()

    while not done:
        _, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=(env.terrain == 99),  # ✅ muestra obstáculos
            active_unit=env._get_active_unit(),
            terrain=env.terrain
        )
        clock.tick(2)  # Ajusta velocidad visual

    pygame.quit()
    print("🎮 Partida finalizada.")

if __name__ == "__main__":
    main()
