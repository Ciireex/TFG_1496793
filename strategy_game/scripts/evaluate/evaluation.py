import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.ppo_mask import MaskablePPO   

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
N_MATCHES = 100
MAX_TURNS = 500  # para evitar partidas infinitas

# === MODELOS DISPONIBLES ===
azules = {
    "a2c_blue": A2C.load(os.path.join(MODEL_DIR, "a2c_blue_f7_v2"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "ppo_blue": PPO.load(os.path.join(MODEL_DIR, "ppo_blue_f7_v4"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "maskableppo_blue": MaskablePPO.load(os.path.join(MODEL_DIR, "maskableppo_blue_f7_v3"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "dqn_blue": DQN.load(os.path.join(MODEL_DIR, "dqn_blue_f7_v3"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
}

rojos = {
    "a2c_red": A2C.load(os.path.join(MODEL_DIR, "a2c_red_f7_v3"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "ppo_red": PPO.load(os.path.join(MODEL_DIR, "ppo_red_f7_v4"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "maskableppo_red": MaskablePPO.load(os.path.join(MODEL_DIR, "maskableppo_red_f7_v3"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
    "dqn_red": DQN.load(os.path.join(MODEL_DIR, "dqn_red_f7_v3"), custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}),
}

# === PARTIDA ENTRE DOS MODELOS ===
def run_match(model_blue, model_red, use_mask_blue=False, use_mask_red=False):
    env = Env_Fase7_Terreno()
    obs, _ = env.reset()
    done = False
    turn_count = 0
    last_total_units = len(env.units)

    while not done:
        mask = env.get_action_mask()
        if env.current_player == 0:
            if use_mask_blue and isinstance(model_blue, MaskablePPO):
                action, _ = model_blue.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model_blue.predict(obs, deterministic=True)
        else:
            if use_mask_red and isinstance(model_red, MaskablePPO):
                action, _ = model_red.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model_red.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        turn_count += 1

        if turn_count >= MAX_TURNS:
            return "draw"

    winner = env.get_winner()  # debe devolver 0 (azul), 1 (rojo), o None si empate
    if winner == 0:
        return "blue"
    elif winner == 1:
        return "red"
    else:
        return "draw"

# === TABLA DE RESULTADOS ===
results = pd.DataFrame(index=azules.keys(), columns=rojos.keys())
results[:] = ""

for blue_name, model_blue in azules.items():
    for red_name, model_red in rojos.items():
        blue_wins = 0
        red_wins = 0
        draws = 0

        print(f"⚔️  {blue_name} vs {red_name}")
        for i in range(N_MATCHES):
            outcome = run_match(
                model_blue, model_red,
                use_mask_blue="maskableppo" in blue_name,
                use_mask_red="maskableppo" in red_name
            )
            if outcome == "blue":
                blue_wins += 1
            elif outcome == "red":
                red_wins += 1
            else:
                draws += 1

        results.loc[blue_name, red_name] = f"{blue_wins}W / {red_wins}L / {draws}D"

# === MOSTRAR TABLA FINAL ===
print("\n📊 RESULTADOS:")
print(results.to_markdown())
results.to_csv(os.path.join(MODEL_DIR, "f7_duels_results.csv"))
