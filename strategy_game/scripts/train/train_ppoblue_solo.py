import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv_V4 import StrategyEnv_V4

# === ENTRENAMIENTO PPO SOLO CAPTURA (v1b) SIN ENEMIGOS ===
print("🏁 Entrenando PPO Azul (ppo_capture_v1c) para capturar sin enemigos...")

env = DummyVecEnv([
    lambda: StrategyEnv_V4(use_obstacles=True, only_blue=True)
])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_capture_v1c/",
    device="cpu"
)

model.learn(
    total_timesteps=500_000,
    tb_log_name="ppo_capture_v1c",
    reset_num_timesteps=True
)

model.save("models/ppo_capture_v1c")
print("\n✅ ENTRENAMIENTO COMPLETADO Y GUARDADO COMO models/ppo_capture_v1c ✅")
