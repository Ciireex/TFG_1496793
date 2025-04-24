import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_strategy.envs.StrategyEnv1vsDummy import StrategyEnv1vsDummy

def make_env():
    def _init():
        return StrategyEnv1vsDummy()
    return _init

if __name__ == "__main__":
    freeze_support()  # Necesario para Windows

    model_path = "ppo_strategy_env1vsdummy_multiinput"
    env = SubprocVecEnv([make_env() for _ in range(8)])

    if os.path.exists(model_path + ".zip"):
        print("Cargando modelo existente con MultiInputPolicy...")
        model = PPO("MultiInputPolicy", env, verbose=1)
        model.set_parameters(model_path + ".zip")
    else:
        print("Creando nuevo modelo con MultiInputPolicy...")
        model = PPO("MultiInputPolicy", env, verbose=1)

    # Entrenamiento extendido con entorno optimizado
    model.learn(total_timesteps=1_000_000)
    model.save(model_path)
    env.close()
