# train_combat.py
import os
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnvCombat import StrategyEnvCombat   # entorno de combate puro

SEED        = 42
NUM_ENVS    = 8
TIMESTEPS   = 1_000_000           # pasos por sesión de entrenamiento
MODEL_NAME  = "ppo_combat_5"
MODEL_PATH  = f"models/{MODEL_NAME}"

def make_env(rank: int):
    """Factory de entornos con semilla reproducible."""
    def _init():
        env = StrategyEnvCombat()
        env.reset(seed=SEED + rank)
        return env
    return _init

if __name__ == "__main__":
    # ── Callbacks: checkpoints cada (250k pasos totales)
    checkpoint_callback = CheckpointCallback(
        save_freq         = 250_000 // NUM_ENVS,
        save_path         = f"./models/{MODEL_NAME}_checkpoints",
        name_prefix       = "chkpt",
        save_vecnormalize = True
    )

    # ── Crea VecEnv paralelo
    envs = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    envs = VecMonitor(envs)                            # registra métricas por episodio
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Fija semillas globales
    np.random.seed(SEED)
    random.seed(SEED)

    # ── Comprueba la conformidad del entorno con la API Gym
    check_env(StrategyEnvCombat(), warn=True)

    # ── Carga modelo existente o crea uno nuevo
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Cargando modelo existente desde {MODEL_PATH}.zip")
        model = PPO.load(
            MODEL_PATH,
            env             = envs,
            tensorboard_log = "./tensorboard_logs",
            device          = "auto",
            custom_objects  = {"learning_rate": 3e-4}
        )
    else:
        print("🚀 Entrenamiento desde cero (combate)")
        model = PPO(
            policy          = "MlpPolicy",
            env             = envs,
            learning_rate   = 3e-4,
            ent_coef        = 0.02,
            clip_range      = 0.2,
            n_steps         = 1024,
            batch_size      = 64,
            n_epochs        = 10,
            verbose         = 1,
            tensorboard_log = "./tensorboard_logs",
            seed            = SEED,
            device          = "auto",
        )

    # ── Entrena (o continúa) por TIMESTEPS pasos
    model.learn(
        total_timesteps = TIMESTEPS,
        callback        = checkpoint_callback
    )

    # ── Guarda el modelo final
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}.zip")

    # ── Guarda el wrapper de normalización
    envs.save(f"models/{MODEL_NAME}_vecnormalize.pkl")
    print(f"✅ VecNormalize guardado en models/{MODEL_NAME}_vecnormalize.pkl")
