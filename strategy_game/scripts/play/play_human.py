import os
import sys
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib.ppo_mask import MaskablePPO

# === CONFIGURACIÓN ===
HUMAN_TEAM = 0  # 0 = azul, 1 = rojo
MODEL_PATH = "dqn_red_f7_v3.zip"

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CARGA DE MODELO RIVAL ===
def load_model(path):
    if "maskableppo" in path.lower():
        return MaskablePPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "ppo" in path.lower():
        return PPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "a2c" in path.lower():
        return A2C.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "dqn" in path.lower():
        return DQN.load(path, device="auto")
    else:
        raise ValueError("Modelo no reconocido")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
model = load_model(os.path.join(MODEL_DIR, MODEL_PATH))

# === ENTORNO Y RENDER ===
env = Env_Fase7_Terreno()
renderer = Renderer(width=1000, height=600, board_size=env.board_size)
obs, _ = env.reset()
done = False
clock = pygame.time.Clock()

print("Controles:")
print("W = ↑  S = ↓  A = ←  D = →  (según la fase)")
print("Q = pasar turno (acción 0)")

# === MAPEADO FINAL ===
KEY_TO_ACTION = {
    pygame.K_w: 3,
    pygame.K_s: 4,
    pygame.K_a: 1,
    pygame.K_d: 2,
    pygame.K_q: 0
}

def get_human_action():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.KEYDOWN and event.key in KEY_TO_ACTION:
                return KEY_TO_ACTION[event.key]

# === BUCLE PRINCIPAL ===
while not done:
    active_unit = env._get_active_unit()
    team = env.current_player

    if team == HUMAN_TEAM:
        print(f"[HUMANO] Fase: {env.phase.upper()}  Pos: {active_unit.position}")
        action = get_human_action()
    else:
        action, _ = model.predict(obs, deterministic=True)

    obs, _, done, _, _ = env.step(action)

    blocked = (env.terrain == 99).astype(np.int8)
    renderer.draw_board(units=env.units,
                        terrain=env.terrain,
                        blocked_positions=blocked,
                        active_unit=active_unit)

    clock.tick(10)

pygame.quit()
