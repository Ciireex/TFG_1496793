import os
import time
import random
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicAgent import HeuristicAgent 

# ⚙️ Configuración
MODEL_PATH = "models/ppo_0"
SLEEP_TIME = 0.4
USE_SEED = False
FIXED_SEED = 1234
RL_TEAM = 0  # Cambia a 1 si quieres que la heurística sea el equipo 0

# 🎮 Cargar modelo entrenado
print("🎮 Cargando modelo RL...")
model = PPO.load(MODEL_PATH)

# 🌍 Crear entorno
env = StrategyEnv()

# 🔁 Semilla
if USE_SEED:
    obs, _ = env.reset(seed=FIXED_SEED)
    print(f"🔁 Semilla usada en reset: {FIXED_SEED}")
else:
    seed = random.randint(0, 9999)
    obs, _ = env.reset(seed=seed)
    print(f"🔁 Semilla usada en reset: {seed}")

# 🧠 Crear agente heurístico
heuristic_agent = HeuristicAgent(team=1 - RL_TEAM)

terminated = False
truncated = False
turn_count = 0

# ⚔️ Simulación por turnos
while not (terminated or truncated):
    current_team = env.current_turn

    if current_team == RL_TEAM:
        action, _ = model.predict(obs, deterministic=False)
    else:
        action = heuristic_agent.get_action(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(SLEEP_TIME)
    turn_count += 1

print("🏁 Partida finalizada.")
