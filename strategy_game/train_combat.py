# train_combat.py
import os, random, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnvCombat import StrategyEnvCombat   # ← nuevo entorno

SEED        = 42
NUM_ENVS    = 8
TIMESTEPS   = 500_000           # Fase 1: combate puro
MODEL_NAME  = "ppo_combat"
MODEL_PATH  = f"models/{MODEL_NAME}"

# ── fábrica de entornos con semilla reproducible ─────────────────────────
def make_env(rank: int):
    def _init():
        env = StrategyEnvCombat()
        env.reset(seed = SEED + rank)
        return env
    return _init

if __name__ == "__main__":
    # Checkpoints cada 250 k pasos totales
    checkpoint_callback = CheckpointCallback(
        save_freq = 250_000 // NUM_ENVS,
        save_path = f"./models/{MODEL_NAME}_checkpoints",
        name_prefix = "chkpt"
    )

    # VecEnv paralelo   (NUM_ENVS sub‑procesos)
    envs = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    envs = VecMonitor(envs)                  # métricas por episodio
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Semillas globales
    np.random.seed(SEED);  random.seed(SEED)

    # Validación rápida del entorno
    check_env(StrategyEnvCombat(), warn=True)

    # Cargar modelo previo o crear uno nuevo
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Cargando modelo {MODEL_PATH}.zip")
        model = PPO.load(
            MODEL_PATH,
            env = envs,
            tensorboard_log = "./tensorboard_logs",
            device = "auto",
            custom_objects = {"learning_rate": 3e-4}
        )
    else:
        print("🚀 Entrenamiento desde cero (combate)")
        model = PPO(
            policy          = "MlpPolicy",
            env             = envs,
            learning_rate   = 3e-4,
            ent_coef        = 0.02,    # entropía un poco > para explorar
            clip_range      = 0.2,
            n_steps         = 1024,    # 128 × 8 envs
            batch_size      = 64,
            n_epochs        = 10,
            verbose         = 1,
            tensorboard_log = "./tensorboard_logs",
            seed            = SEED,
            device          = "auto",
        )

    # Entrenamiento
    model.learn(
        total_timesteps = TIMESTEPS,
        callback        = checkpoint_callback
    )

    # Guardado final
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}.zip")
