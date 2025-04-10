import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from gym_strategy.envs.StrategyEnv import StrategyEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Crear entorno vectorizado
def make_env():
    def _init():
        env = StrategyEnv()
        return env
    return _init

if __name__ == "__main__":
    NUM_ENVS = 8
    TIMESTEPS = 1_000_000
    MODEL_NAME = "ppo_0"
    MODEL_PATH = f"models/{MODEL_NAME}"

    # Checkpoints automáticos
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // NUM_ENVS,
        save_path=f"./models/{MODEL_NAME}_checkpoints",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Vectorizar entornos
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Verificar entorno
    check_env(StrategyEnv(), warn=True)

    # Cargar o crear modelo
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f" Cargando modelo desde {MODEL_PATH}.zip")
        model = PPO.load(
            MODEL_PATH,
            tensorboard_log="./tensorboard_logs",
            device="auto"
        )
        model.set_env(env)
    else:
        print("Entrenando desde cero")
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="./tensorboard_logs",
            device="auto"
        )

    # Entrenar
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=checkpoint_callback
    )

    # Guardar modelo final
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}.zip")
