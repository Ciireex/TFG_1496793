import os
import sys
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1 import StrategyEnv_TransferSmall_1v1
from gym_strategy.utils.CustomCNN import CustomCNN

### === WRAPPERS === ###

# Azul aleatorio: moverse o atacar si puede
class RandomBlueWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        if self.env.current_player == 1:  # RED (modelo) juega
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:  # BLUE (random)
            obs_space = self.env.action_space
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, info = self.env.step(action)
            reward = 0.0  # No recompensa para azul
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = 1  # Empieza RED (modelo)
        return obs, info

# Red entrenado vs azul (no aprende)
class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})

    def step(self, action):
        if self.env.current_player == 0:  # BLUE (modelo entrena)
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:  # RED (modelo fijo)
            obs_tensor = torch.tensor(self.env._get_obs()).unsqueeze(0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(int(model_action))
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = 0  # Empieza BLUE
        return obs, info


### === FUNCIONES DE ENTRENAMIENTO === ###

def train_model(env_fn, log_dir, model_name):
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    eval_env = DummyVecEnv([env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "/best_model",
                                 log_path=log_dir + "/eval", eval_freq=5000, deterministic=True)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir + "/checkpoints",
                                             name_prefix=model_name)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=500_000, callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(log_dir, "final_model"))
    print(f"✅ Entrenamiento completado: {model_name}")
    return os.path.join(log_dir, "final_model")


### === ENTRENAMIENTO SECUENCIAL === ###

# 1. Entrenar al equipo ROJO contra un azul aleatorio
print("\n=== ENTRENANDO ROJO VS AZUL ALEATORIO ===")
model_red_path = train_model(
    env_fn=lambda: RandomBlueWrapper(StrategyEnv_TransferSmall_1v1()),
    log_dir="./logs/ppo_red_vs_random",
    model_name="ppo_red2x"
)

# 2. Entrenar al equipo AZUL contra el modelo ROJO ya entrenado
print("\n=== ENTRENANDO AZUL VS ROJO FIJO ===")
train_model(
    env_fn=lambda: FixedRedWrapper(StrategyEnv_TransferSmall_1v1(), model_red_path),
    log_dir="./logs/ppo_blue_vs_trained_red",
    model_name="ppo_blue2x"
)
