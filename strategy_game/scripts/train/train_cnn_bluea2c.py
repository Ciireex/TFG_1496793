import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_strategy.envs.StrategyEnv_CNN import StrategyEnv_CNN
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy
import torch
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === WRAPPER PARA A2C VS HEURÍSTICA ===
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
        print(f"[RESET] Episodio #{self.episode_count} - Controlando equipo {self.controlled_team} (0 = Azul, 1 = Rojo)")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

# === FACTORÍA PARA ENTORNOS ===
def make_env():
    return lambda: DualTeamEnvWrapper(StrategyEnv_CNN(use_obstacles=True), controlled_team=0)

# === MAIN TRAINING BLOCK ===
if __name__ == "__main__":
    print("🏁 Entrenando A2C-CNN (azul) vs Heurística (rojo) en StrategyEnv_CNN...")

    train_env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])

    model = A2C(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./logs/a2c_cnn_blue/",
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256)
        ),
        n_steps=5,  # Por defecto en A2C, puedes ajustar si quieres
        learning_rate=7e-4,  # Por defecto, ajustable
    )

    model.learn(
        total_timesteps=2_000_000,
        tb_log_name="a2c_cnn_blue",
        reset_num_timesteps=True,
        callback=CheckpointCallback(save_freq=100_000, save_path="./models/", name_prefix="a2c_cnn_blue")
    )

    model.save("models/a2c_cnn_blue")
    print("\n✅ ENTRENAMIENTO COMPLETADO Y GUARDADO ✅")
