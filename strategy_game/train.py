import os
import sys

# Asegura que el path de gym_strategy esté accesible
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from gym_strategy.envs.StrategyEnv import StrategyEnv

def make_env():
    def _init():
        env = StrategyEnv()
        return env
    return _init

if __name__ == "__main__":
    NUM_ENVS = 8
    TIMESTEPS = 1_000_000
    MODEL_NAME = "ppo_00"
    MODEL_PATH = f"models/{MODEL_NAME}"

    # Multiproceso
    envs = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Verificación del entorno
    check_env(StrategyEnv(), warn=True)

    # Cargar modelo previo o crear uno nuevo
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Cargando modelo existente desde {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=envs, tensorboard_log="./tensorboard_logs", device="auto")
    else:
        print("🚀 Entrenamiento desde cero (ppo_0).")
        model = PPO("MlpPolicy", env=envs, verbose=1, tensorboard_log="./tensorboard_logs", device="auto")

    # Entrenar
    model.learn(total_timesteps=TIMESTEPS)

    # Guardar
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}.zip")
