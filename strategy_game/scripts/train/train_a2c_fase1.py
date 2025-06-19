import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from gym_strategy.envs.StrategyEnv_Fase3 import StrategyEnv_Fase3
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === WRAPPERS ===

class DummyRedWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 0:
            return self.env.step(action)
        else:
            return self.env.step(0)

class DummyBlueWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 1:
            return self.env.step(action)
        else:
            return self.env.step(0)

# === FUNC. ENTRENAMIENTO ===

def train_a2c(env_fn, save_path, total_timesteps=300_000):
    env = DummyVecEnv([env_fn])
    model = A2C(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/tb_a2c_fase3",
        policy_kwargs=dict(features_extractor_class=CustomCNN)
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

# === ENTRENAR AZUL VS ROJO DUMMY ===
model_path_blue = "./logs/a2c_blue_fase3_nobstacles/final_model"
os.makedirs(os.path.dirname(model_path_blue), exist_ok=True)

train_a2c(
    env_fn=lambda: DummyRedWrapper(StrategyEnv_Fase3(obstacle_count=0)),
    save_path=model_path_blue
)

# === ENTRENAR ROJO VS AZUL DUMMY ===
model_path_red = "./logs/a2c_red_fase3_nobstacles/final_model"
os.makedirs(os.path.dirname(model_path_red), exist_ok=True)

train_a2c(
    env_fn=lambda: DummyBlueWrapper(StrategyEnv_Fase3(obstacle_count=0)),
    save_path=model_path_red
)
