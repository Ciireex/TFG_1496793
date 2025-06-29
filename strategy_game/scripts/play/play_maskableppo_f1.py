import os
import sys
import pygame
from sb3_contrib.ppo_mask import MaskablePPO

# Añadir ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.core.Renderer import Renderer

# === CONFIGURACIÓN ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_blue_f1_v1.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_red_f1_v1.zip")

# === CARGA DE MODELOS ===
print("🧠 Cargando modelos MaskablePPO F1...")
model_blue = MaskablePPO.load(BLUE_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
model_red = MaskablePPO.load(RED_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
print("✅ Modelos cargados correctamente.")

# === ENVOLTORIO DE ENTORNO PARA PLAY ===
class DualMaskableEnv(Env_Fase1_Soldiers4x4):
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

# === MAIN LOOP ===
def main():
    pygame.init()
    env = DualMaskableEnv()
    obs, _ = env.reset()
    renderer = Renderer(width=600, height=600, board_size=env.board_size)

    done = False
    clock = pygame.time.Clock()

    while not done:
        _, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=getattr(env, "blocked_positions", None),
            active_unit=env._get_active_unit(),
            terrain=env.terrain
        )
        clock.tick(2)

    pygame.quit()
    print("🎮 Partida finalizada.")

if __name__ == "__main__":
    main()
