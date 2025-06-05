import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv_V5 import StrategyEnv_V5
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# === CUSTOM CNN EXTRACTOR para obs de (16, 7, 5) ===
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # → (32, 7, 5)
            nn.ReLU(),
            nn.Flatten(),  # → 32×7×5 = 1120
        )
        self.linear = nn.Sequential(
            nn.Linear(1120, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# Wrapper para que PPO (azul) juegue contra heurística (rojo)
class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.heuristic = HeuristicPolicy(base_env)
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        print(f"[RESET] Episodio #{self.episode_count}")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

# === CONFIGURACIÓN DEL ENTORNO Y RED ===
print("🏁 Entrenando PPO-CNN (azul) vs Heurística (rojo) en StrategyEnv_V5...")

env = DummyVecEnv([
    lambda: DualTeamEnvWrapper(StrategyEnv_V5(use_obstacles=True), controlled_team=0)
])

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64)
)

model = PPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/ppo_cnn_vs_heuristic_v10/",
    device="cuda" if th.cuda.is_available() else "cpu"
)

model.learn(
    total_timesteps=1_000_000,
    tb_log_name="ppo_cnn_vs_heuristic_v10",
    reset_num_timesteps=True
)

model.save("models/ppo_cnn_vs_heuristic_v10")
print("\n✅ ENTRENAMIENTO CNN COMPLETADO Y GUARDADO ✅")
