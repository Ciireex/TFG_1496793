# play_simple_minimal.py

import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvSimple import StrategyEnvSimple

MODEL_PATH = "models/ppo_simple_4.zip"
SLEEP_TIME = 0.5

# 1) Entorno directo
env = StrategyEnvSimple()

# 2) Modelo
model = PPO.load(MODEL_PATH)

# 3) Reset inicial (Gymnasium style devuelve obs, info)
obs, info = env.reset(seed=None, options=None)

terminated = False
truncated  = False

action_names = ["Mover", "Atacar", "Capturar", "Pasar"]

while not (terminated or truncated):
    # 4) Política
    action, _ = model.predict(obs, deterministic=True)
    act, tx, ty = action

    # 5) Mostrar
    print(f"Equipo {env.current_turn} → {action_names[act]} → destino ({tx},{ty})")

    # 6) step puro (5‑tupla)
    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(SLEEP_TIME)

print("🏁 Partida finalizada.")
