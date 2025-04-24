import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_strategy.envs.StrategyEnv1vs1 import StrategyEnv1vs1

def make_env():
    def _init():
        return StrategyEnv1vs1()
    return _init

if __name__ == "__main__":
    freeze_support()  # Necesario en Windows si usas SubprocVecEnv

    model_path = "ppo_strategy_env1vs1_v4"
    env = SubprocVecEnv([make_env() for _ in range(8)])

    if os.path.exists(model_path + ".zip"):
        print("Cargando modelo existente...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creando nuevo modelo...")
        model = PPO("MlpPolicy", env, verbose=1)

    # Entrena con 500k timesteps (ajusta según necesites)
    model.learn(total_timesteps=500_000)
    model.save(model_path)
    env.close()
