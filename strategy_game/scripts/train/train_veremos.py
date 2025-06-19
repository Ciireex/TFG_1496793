
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
import torch
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === WRAPPER PARA CONTROLAR CADA EQUIPO POR SEPARADO ===
class TeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, team_id, opponent_policy):
        super().__init__(base_env)
        self.team_id = team_id
        self.opponent_policy = opponent_policy
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.team_id:
            action, _ = self.opponent_policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.team_id:
            action, _ = self.opponent_policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

# === ENTRENAMIENTO ===
if __name__ == "__main__":
    print("🏁 Entrenando PPO Azul vs PPO Rojo (self-play alternado) en CastleEnv...")

    # Modelo dummy para empezar entrenamiento del primero
    dummy_env = StrategyEnv_Castle()
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    model_red = PPO("CnnPolicy", dummy_vec, verbose=0, device="cpu", policy_kwargs=dict(
        features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256)
    ))
    model_red.save("models/ppo_castle_red")

    # Entrenar modelo azul contra rojo
    env_blue = DummyVecEnv([lambda: TeamEnvWrapper(StrategyEnv_Castle(), team_id=0, opponent_policy=model_red)])
    model_blue = PPO("CnnPolicy", env_blue, verbose=1, device="cpu", tensorboard_log="./logs/ppo_blue/",
        policy_kwargs=dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256)),
        n_steps=1024, batch_size=256, learning_rate=2.5e-4)
    model_blue.learn(total_timesteps=1_000_000, callback=CheckpointCallback(save_freq=100_000, save_path="./models/", name_prefix="ppo_castle_blue"))
    model_blue.save("models/ppo_castle_blue")

    # Entrenar modelo rojo contra azul entrenado
    env_red = DummyVecEnv([lambda: TeamEnvWrapper(StrategyEnv_Castle(), team_id=1, opponent_policy=model_blue)])
    model_red = PPO("CnnPolicy", env_red, verbose=1, device="cpu", tensorboard_log="./logs/ppo_red/",
        policy_kwargs=dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256)),
        n_steps=1024, batch_size=256, learning_rate=2.5e-4)
    model_red.learn(total_timesteps=1_000_000, callback=CheckpointCallback(save_freq=100_000, save_path="./models/", name_prefix="ppo_castle_red"))
    model_red.save("models/ppo_castle_red")

    print("\n✅ ENTRENAMIENTO COMPLETADO Y MODELOS GUARDADOS ✅")
