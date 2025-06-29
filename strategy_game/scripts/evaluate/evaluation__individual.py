import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import A2C

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
N_MATCHES = 100
MAX_TURNS = 500

# === CARGA DE MODELOS A2C ===
model_blue = A2C.load(
    os.path.join(MODEL_DIR, "a2c_blue_f7_v2"),
    custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}
)
model_red = A2C.load(
    os.path.join(MODEL_DIR, "a2c_red_f7_v3"),
    custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}
)

# === FUNCIÓN DE PARTIDA ===
def run_match():
    env = Env_Fase7_Terreno()
    obs, _ = env.reset()
    done = False
    turn_count = 0

    while not done:
        if env.current_player == 0:
            action, _ = model_blue.predict(obs, deterministic=True)
        else:
            action, _ = model_red.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        turn_count += 1

        if turn_count >= MAX_TURNS:
            return "draw"

    winner = env.get_winner()
    if winner is None:
        alive_teams = set(u.team for u in env.units if u.is_alive())
        if len(alive_teams) == 1:
            winner = list(alive_teams)[0]
    return "blue" if winner == 0 else "red" if winner == 1 else "draw"

# === EJECUCIÓN ===
print("⚔️  Evaluando A2C Azul vs A2C Rojo (F7_v2 vs F7_v3)...")
blue_wins = 0
red_wins = 0
draws = 0

for i in range(N_MATCHES):
    result = run_match()
    if result == "blue":
        blue_wins += 1
    elif result == "red":
        red_wins += 1
    else:
        draws += 1

# === RESULTADO FINAL ===
summary = f"{blue_wins}W / {red_wins}L / {draws}D"
print("\n📊 RESULTADO FINAL:")
print(f"A2C Azul vs Rojo: {summary}")

# === GUARDAR EN CSV ===
result_df = pd.DataFrame(
    [[summary]],
    index=["dqn_blue_f7_v3"],
    columns=["dqn_red_f7_v3"]
)
result_df.to_csv(os.path.join(MODEL_DIR, "dqn_vs_dqn_f7_v3.csv"))
