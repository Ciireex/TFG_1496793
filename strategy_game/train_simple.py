import os, random, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnvSimple import StrategyEnvSimple

SEED        = 42
NUM_ENVS    = 8
TIMESTEPS   = 1_000_000
MODEL_NAME  = "ppo_captura"
MODEL_PATH  = f"models/{MODEL_NAME}"

# ─── fábrica de entornos con semillas reproducibles ──────────────
def make_env(rank: int):
    def _init():
        env = StrategyEnvSimple()
        env.reset(seed=SEED + rank)          # semilla distinta por sub‑proceso
        return env
    return _init

if __name__ == "__main__":
    # Call‑back de checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq = 250_000 // NUM_ENVS,
        save_path = f"./models/{MODEL_NAME}_checkpoints",
        name_prefix = "chkpt",
        save_vecnormalize = True
    )

    # VecEnv paralelo
    envs = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Semillas globales
    np.random.seed(SEED);  random.seed(SEED)

    # Validación rápida del entorno
    check_env(StrategyEnvSimple(), warn=True)

    # Cargar o crear modelo
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Cargando {MODEL_PATH}.zip")
        model = PPO.load(
            MODEL_PATH,
            env = envs,
            tensorboard_log = "./tensorboard_logs",
            device = "auto",
            custom_objects = {"learning_rate": 3e-4}
        )
    else:
        print("🚀 Entrenamiento desde cero")
        model = PPO(
            "MlpPolicy",
            env = envs,
            learning_rate = 3e-4,
            ent_coef = 0.01,
            clip_range = 0.2,
            n_steps = 1024,              # 128 steps/env
            batch_size = 64,
            n_epochs = 10,
            verbose = 1,
            tensorboard_log = "./tensorboard_logs",
            seed = SEED,
            device = "auto",
        )

    # Entrenar
    model.learn(total_timesteps = TIMESTEPS,
                callback = checkpoint_callback)

    # Guardar
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}.zip")
