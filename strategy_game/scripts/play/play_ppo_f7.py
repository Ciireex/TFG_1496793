import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.core.Renderer import Renderer

# === MODELOS ===
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f7_v4.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_red_f7_v4.zip")

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === FUNCIONES ===
def load_fc_layers_only(model: PPO, path: str):
    data, params, _ = load_from_zip_file(path)
    model_state = model.policy.state_dict()
    filtered_state = {
        k: v for k, v in params.items()
        if k in model_state and model_state[k].shape == v.shape and not k.startswith("features_extractor")
    }
    model.policy.load_state_dict(filtered_state, strict=False)
    print("✅ Pesos pi/vf cargados para modelo congelado.")

# === ENTORNO Y RENDER ===
env = Env_Fase7_Terreno()
renderer = Renderer(width=900, height=540, board_size=env.board_size)

# === MODELO AZUL ===
blue_model = PPO.load(BLUE_MODEL_PATH, env=env, device="cpu")

# === MODELO ROJO RECONSTRUIDO ===
print("🧊 Adaptando modelo rojo congelado F6 al tamaño del entorno F7...")
red_model = PPO(policy="CnnPolicy", env=env, policy_kwargs=policy_kwargs, device="cpu", verbose=0)
load_fc_layers_only(red_model, RED_MODEL_PATH)

# === INICIAR PARTIDA ===
obs, _ = env.reset()
done = False
turn_count = 0

while not done:
    # === Construir obstáculos manualmente ===
    blocked = np.zeros(env.board_size, dtype=np.uint8)
    for x in range(env.board_size[0]):
        for y in range(env.board_size[1]):
            if env.terrain[x, y] == 99:
                blocked[x, y] = 1

    # === Obtener unidad activa de forma segura ===
    active_unit = env._get_active_unit() if hasattr(env, "_get_active_unit") else None

    # === Render ===
    renderer.draw_board(
        units=env.units,
        blocked_positions=blocked,
        active_unit=active_unit,
        terrain=getattr(env, "terrain_layer", env.terrain),
        highlight_attack=True
    )
    pygame.time.delay(250)

    # === Acción IA ===
    if env.current_player == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
    else:
        action, _ = red_model.predict(obs, deterministic=True)

    obs, reward, done, _, _ = env.step(action)
    turn_count += 1

# === Render final ===
renderer.draw_board(
    units=env.units,
    blocked_positions=blocked,
    active_unit=None,
    terrain=getattr(env, "terrain_layer", env.terrain),
    highlight_attack=False
)
pygame.time.wait(500)
pygame.quit()

# === Resultado textual ===
print(f"\n🏁 Partida terminada en {turn_count} turnos.")
if reward == 1:
    print("🔵 Victoria del EQUIPO AZUL")
elif reward == -1:
    print("🔴 Victoria del EQUIPO ROJO")
else:
    print("⚪ Empate o final sin ganador")
