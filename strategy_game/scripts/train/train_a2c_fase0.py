import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from gym_strategy.envs.StrategyEnv_TransferMedium_1v1_Archers import StrategyEnv_TransferMedium_1v1_Archers
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === WRAPPERS ===

class DummyRedWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 0:
            return self.env.step(action)
        else:
            return self.env.step(0)  # Acción 0 = no moverse

class FixedBlueWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = A2C.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model.policy.eval()

    def step(self, action):
        if self.env.current_player == 1:
            return self.env.step(action)
        else:
            obs = self.env._get_obs()
            obs_tensor = th.tensor(obs).unsqueeze(0)
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            return self.env.step(int(action))

# === ENTRENADOR ===

def train_a2c(env_fn, model_path, total_timesteps=100_000):
    env = DummyVecEnv([env_fn])
    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs={"features_extractor_class": CustomCNN}
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model

# === FASE 1: entrenar azul contra red tonto ===
model_path_blue = "./logs/a2c_blue_vs_dummy_red_fase1/final_model"
os.makedirs(os.path.dirname(model_path_blue), exist_ok=True)

train_a2c(lambda: DummyRedWrapper(StrategyEnv_TransferMedium_1v1_Archers()), model_path_blue)

# === FASE 2: entrenar rojo contra azul fijo ===
model_path_red = "./logs/a2c_red_vs_fixed_blue_fase2/final_model"
os.makedirs(os.path.dirname(model_path_red), exist_ok=True)

train_a2c(lambda: FixedBlueWrapper(StrategyEnv_TransferMedium_1v1_Archers(), model_path_blue), model_path_red)
