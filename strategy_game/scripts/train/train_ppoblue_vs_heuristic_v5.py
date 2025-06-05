import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv_V4 import StrategyEnv_V4

# === ENTRENAMIENTO PPO AZUL SOLO CAPTURA ===
print("🏁 Entrenando PPO Azul (v6) para capturar sin enemigos...")

env = DummyVecEnv([
    lambda: StrategyEnv_V4(use_obstacles=True, only_blue=True)  # Solo equipo azul
])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppoblue_captura_solo_v6/",
    device="cpu"
)

model.learn(
    total_timesteps=300_000,
    tb_log_name="PPO_captura_solo_v6",
    reset_num_timesteps=True
)

# Guardar modelo final
model.save("models/ppoblue_captura_solo_v6")

print("\n✅ ENTRENAMIENTO SOLO CAPTURA COMPLETADO Y GUARDADO COMO ppoblue_captura_solo_v6 ✅")
