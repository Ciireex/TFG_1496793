import os
import time
import random
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvSA import StrategyEnvSA  # Usamos el entorno actualizado con debilidades

# Configuración
MODEL_PATH_0 = "models/ppo_sa"  # Modelo del equipo 0
MODEL_PATH_1 = "models/ppo_sa"  # Modelo del equipo 1 
SLEEP_TIME = 0.4                # Tiempo de espera entre turnos para poder seguir la partida visualmente
USE_SEED = False                # Aleatoriedad de posición inicial o no
FIXED_SEED = 1234              # Semilla fija 

# Cargamos modelos 
print("Cargando modelos...")
model_0 = PPO.load(MODEL_PATH_0)
model_1 = PPO.load(MODEL_PATH_1)

# Creamos el entorno
env = StrategyEnvSA()

# Inicializamos el entorno con semilla fija o aleatoria
if USE_SEED:
    obs, _ = env.reset(seed=FIXED_SEED)
    print(f"Semilla usada en reset: {FIXED_SEED}")
else:
    seed = random.randint(0, 9999)
    obs, _ = env.reset(seed=seed)
    print(f"Semilla usada en reset: {seed}")

# Variables de control del bucle
terminated = False
truncated = False
turn_count = 0

# Bucle de juego por turnos hasta que alguien gane o se agote el tiempo
while not (terminated or truncated):
    current_team = env.current_turn

    if current_team == 0:
        action, _ = model_0.predict(obs, deterministic=False)
    else:
        action, _ = model_1.predict(obs, deterministic=False)

    obs, reward, terminated, truncated, info = env.step(action) # Ejecutamos la acción y obtenemos la nueva observación y el estado del entorno

    time.sleep(SLEEP_TIME)
    turn_count += 1

print("!!Partida finalizada.")
