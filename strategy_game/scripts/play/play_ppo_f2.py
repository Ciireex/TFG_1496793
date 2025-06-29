import os
import sys
import pygame
from stable_baselines3 import PPO

# Añadir ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase2_Soldiers6x4
from gym_strategy.core.Renderer import Renderer

# === CARGA DE MODELOS PPO ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
blue_model = PPO.load(os.path.join(MODEL_DIR, "ppo_blue_f2_v3.zip"), device="auto")
red_model = PPO.load(os.path.join(MODEL_DIR, "ppo_red_f2_v3.zip"), device="auto")

# === ENTORNO Y RENDER ===
env = Env_Fase2_Soldiers6x4()
renderer = Renderer(width=700, height=500, board_size=env.board_size)
obs, _ = env.reset()
done = False

# === BUCLE DE JUEGO ===
while not done:
    renderer.draw_board(
        units=env.units,
        active_unit=env._get_active_unit(),
        terrain=getattr(env, "terrain_map", None)  # opcional, si tienes terreno
        # blocked_positions=getattr(env, "blocked_positions", None),  # ← Comentado
    )

    if env.current_player == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    obs, _, done, _, _ = env.step(action)
    pygame.time.delay(300)

pygame.quit()
