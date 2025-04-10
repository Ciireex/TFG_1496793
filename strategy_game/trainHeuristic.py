import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnvHeuristic import StrategyEnvHeuristic

if __name__ == "__main__":
    TIMESTEPS = 1_000_000
    MODEL_NAME = "ppo_vs_heuristic"
    MODEL_PATH = f"models/{MODEL_NAME}"

    # Checkpoints automáticos
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # Frecuencia de guardado
        save_path=f"./models/{MODEL_NAME}_checkpoints",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Entorno no paralelo (solo se entrena el equipo 0)
    env = DummyVecEnv([lambda: StrategyEnvHeuristic()])

    # Verificación de entorno (opcional)
    check_env(StrategyEnvHeuristic(), warn=True)

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
        print("Entrenando desde cero contra IA heurística")
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
    print(f" Modelo guardado en {MODEL_PATH}.zip")
