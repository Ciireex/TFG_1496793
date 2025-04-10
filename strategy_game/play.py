import os
import time
import random
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv

# Configuración
MODEL_PATH_0 = "models/ppo_0"
MODEL_PATH_1 = "models/ppo_0"  
SLEEP_TIME = 0.4  # segundos por paso
USE_SEED = False 
FIXED_SEED = 1234  

# Cargar modelos
print("🎮 Cargando modelos...")
model_0 = PPO.load(MODEL_PATH_0)
model_1 = PPO.load(MODEL_PATH_1)

# Crear entorno
env = StrategyEnv()

# Semilla aleatoria o fija
if USE_SEED:
    obs, _ = env.reset(seed=FIXED_SEED)
    print(f"Semilla usada en reset: {FIXED_SEED}")
else:
    seed = random.randint(0, 9999)
    obs, _ = env.reset(seed=seed)
    print(f"Semilla usada en reset: {seed}")

terminated = False
truncated = False
turn_count = 0

# Simulación por turnos
while not (terminated or truncated):
    current_team = env.current_turn

    if current_team == 0:
        action, _ = model_0.predict(obs, deterministic=False)
    else:
        action, _ = model_1.predict(obs, deterministic=False)

    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(SLEEP_TIME)
    turn_count += 1

print("Partida finalizada.")
