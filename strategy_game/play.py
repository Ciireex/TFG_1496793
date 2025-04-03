import os
import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv

# ⚙️ Configuración
MODEL_PATH_0 = "models/ppo_00"  # 👈 Modelo del equipo 0
MODEL_PATH_1 = "models/ppo_00"  # 👈 Modelo del equipo 1
SLEEP_TIME = 0.5  # Tiempo entre turnos (segundos)

# Cargar modelos
print("🎮 Cargando modelos...")
model_0 = PPO.load(MODEL_PATH_0)
model_1 = PPO.load(MODEL_PATH_1)

# Crear entorno y resetear
env = StrategyEnv()
obs, _ = env.reset()

terminated = False
truncated = False
turn_count = 0

# ⚔️ Juego por turnos entre dos modelos
while not (terminated or truncated):
    current_team = env.current_turn

    if current_team == 0:
        action, _ = model_0.predict(obs, deterministic=True)
    else:
        action, _ = model_1.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(SLEEP_TIME)
    turn_count += 1

print("🏁 Partida finalizada.")
