import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from gym_strategy.envs.StrategyEnv_Fase3 import StrategyEnv_Fase3
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === WRAPPERS ===

class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = A2C.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model.policy.eval()

    def step(self, action):
        if self.env.current_player == 0:
            return self.env.step(action)
        else:
            obs = self.env._get_obs()
            obs_tensor = th.tensor(obs).unsqueeze(0)
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            return self.env.step(int(action))


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

# === FUNC. ENTRENAMIENTO ===

def train_a2c(env_fn, save_path, total_timesteps=300_000):
    env = DummyVecEnv([env_fn])
    model = A2C(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/tb_a2c_fase4",
        policy_kwargs=dict(features_extractor_class=CustomCNN)
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

# === ENTRENAR AZUL VS ROJO CONGELADO (obstáculos) ===
model_path_red_fijo = "./logs/a2c_red_fase3_nobstacles/final_model"
model_path_blue_final = "./logs/a2c_blue_fase4_obstacles/final_model"
os.makedirs(os.path.dirname(model_path_blue_final), exist_ok=True)

train_a2c(
    env_fn=lambda: FixedRedWrapper(StrategyEnv_Fase3(obstacle_count=10), model_path_red_fijo),
    save_path=model_path_blue_final
)

# === ENTRENAR ROJO VS AZUL CONGELADO (obstáculos) ===
model_path_blue_fijo = model_path_blue_final
model_path_red_final = "./logs/a2c_red_fase4_obstacles/final_model"
os.makedirs(os.path.dirname(model_path_red_final), exist_ok=True)

train_a2c(
    env_fn=lambda: FixedBlueWrapper(StrategyEnv_Fase3(obstacle_count=10), model_path_blue_fijo),
    save_path=model_path_red_final
)
