import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase7_MapaGrande
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.core.Renderer import Renderer

# === CONFIGURACIÓN DE MODELOS ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f7_v3.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_red_f7_v1.zip")

# === CARGA DE MODELOS ENTRENADOS EN F7 ===
print("🧠 Cargando modelos PPO (azul) vs MaskablePPO (rojo)...")
model_blue = PPO.load(BLUE_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
model_red = MaskablePPO.load(RED_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})
print("✅ Modelos cargados correctamente.")

# === ENVOLTORIO PARA ENFRENTAR PPO AZUL VS MASKABLEPPO ROJO ===
class EnvPPOvsMaskable(Env_Fase7_MapaGrande):
    def __init__(self):
        super().__init__()

    def step(self, _):
        obs = self._get_obs()
        mask = self.get_action_mask()
        if self.current_player == 0:
            action, _ = model_blue.predict(obs, deterministic=True)  # PPO normal (sin máscara)
        else:
            action, _ = model_red.predict(obs, action_masks=mask, deterministic=True)
        return super().step(action)

# === LOOP PRINCIPAL DE VISUALIZACIÓN ===
def main():
    pygame.init()
    env = EnvPPOvsMaskable()
    obs, _ = env.reset()
    renderer = Renderer(width=1000, height=600, board_size=env.board_size)

    done = False
    clock = pygame.time.Clock()

    while not done:
        _, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            blocked_positions=(env.terrain == 99),
            active_unit=env._get_active_unit(),
            terrain=env.terrain
        )
        clock.tick(10)

    pygame.quit()

    # === ANUNCIAR GANADOR ===
    winner = env.get_winner()
    if winner == 0:
        print("🏆 ¡Ha ganado el equipo AZUL (PPO)!")
    elif winner == 1:
        print("🏆 ¡Ha ganado el equipo ROJO (MaskablePPO)!")
    else:
        print("🤝 La partida ha terminado en EMPATE.")

if __name__ == "__main__":
    main()
