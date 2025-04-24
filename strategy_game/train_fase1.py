import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_strategy.envs.StrategyEnvFase1 import StrategyEnvFase1

# Configuración
NUM_ENVS = 8
TIMESTEPS = 500_000
MODEL_NAME = "ppo_fase1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Función de creación de entornos
def make_env():
    def _init():
        env = StrategyEnvFase1()
        env.use_knights = False
        env.use_archers = False
        env.use_capture = False
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    envs = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Cargando modelo existente desde {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=envs, tensorboard_log="./tensorboard_logs", device="auto")
    else:
        print("🚀 Entrenamiento desde cero (Fase 1: soldados)")
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./tensorboard_logs", device="auto")

    model.learn(total_timesteps=TIMESTEPS)

    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}.zip")
