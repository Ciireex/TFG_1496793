import os
import sys
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym_strategy.envs.StrategyEnv_TransferSmall_1v1 import StrategyEnv_TransferSmall_1v1
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# Wrapper para que PPO solo controle un equipo
class RandomEnemyWrapper(gym.Wrapper):
    def __init__(self, env, team_controlled=0):
        super().__init__(env)
        self.team_controlled = team_controlled

    def step(self, action):
        if self.env.current_player == self.team_controlled:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            random_action = random.randint(0, self.action_space.n - 1)
            obs, _, terminated, truncated, info = self.env.step(random_action)
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = self.team_controlled
        return obs, info

# Política personalizada
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

def train_team(team_id, log_dir):
    print(f"\n--- Entrenando equipo {'AZUL' if team_id == 0 else 'ROJO'} ---\n")

    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        base_env = StrategyEnv_TransferSmall_1v1()
        return RandomEnemyWrapper(base_env, team_controlled=team_id)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "/best_model",
                                 log_path=log_dir + "/eval", eval_freq=5000, deterministic=True)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir + "/checkpoints",
                                             name_prefix=f"ppo_team{team_id}")

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

    model.learn(total_timesteps=300_000, callback=[eval_callback, checkpoint_callback])

    model.save(log_dir + "/final_model")
    print(f"\n✅ Entrenamiento equipo {team_id} completado y guardado en {log_dir}/final_model\n")

# Entrenar azul vs random
train_team(team_id=0, log_dir="./logs/ppo_1v1_blue_vs_random")

# Entrenar rojo vs random
train_team(team_id=1, log_dir="./logs/ppo_1v1_red_vs_random")
