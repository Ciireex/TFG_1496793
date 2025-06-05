import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

from gym_strategy.envs.StrategyEnv import StrategyEnv

# === Wrapper para enfrentamiento DQN vs DQN con acción aplanada ===
class DualTeamVsModelWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team, opponent_model):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.opponent_model = opponent_model
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            obs_flat = obs.flatten().reshape(1, -1)
            action, _ = self.opponent_model.predict(obs_flat, deterministic=True)
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        return obs, info

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            obs_flat = obs.flatten().reshape(1, -1)
            action_op, _ = self.opponent_model.predict(obs_flat, deterministic=True)
            if isinstance(action_op, (np.ndarray, list)):
                action_op = int(action_op[0])
            obs, _, terminated, truncated, info = self.env.step(action_op)
        return obs, reward, terminated, truncated, info

# === Crea entorno compatible con DQN y wrapper de duelo ===
def make_env(controlled_team, opponent_model):
    return DummyVecEnv([
        lambda: FlattenObservation(
            DualTeamVsModelWrapper(StrategyEnv(use_obstacles=True), controlled_team, opponent_model)
        )
    ])

# === Cargar modelos base ===
blue_model = DQN.load("models/dqn_blue_vs_heuristic_v2", device="cpu")
red_model = DQN.load("models/dqn_red_vs_heuristic_v2", device="cpu")

# === Entrenamiento cíclico v4 ===
for ciclo in range(1, 11):
    print(f"\n🌀 CICLO {ciclo} / 10")

    # Entrenar Azul contra Rojo
    print("🔵 Entrenando DQN Azul v4 contra Rojo...")
    env_blue = make_env(0, red_model)
    blue_model = DQN(
        "MlpPolicy",
        env_blue,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="logs/dqn_blue_v4/",
        device="cpu"
    )
    blue_model.learn(total_timesteps=100_000, reset_num_timesteps=False)
    path_blue = f"models/dqn_blue_v4_cycle{ciclo}.zip"
    blue_model.save(path_blue)
    print(f"💾 Guardado: {path_blue}")

    # Entrenar Rojo contra nuevo Azul
    print("🔴 Entrenando DQN Rojo v4 contra Azul...")
    env_red = make_env(1, blue_model)
    red_model = DQN(
        "MlpPolicy",
        env_red,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="logs/dqn_red_v4/",
        device="cpu"
    )
    red_model.learn(total_timesteps=100_000, reset_num_timesteps=False)
    path_red = f"models/dqn_red_v4_cycle{ciclo}.zip"
    red_model.save(path_red)
    print(f"💾 Guardado: {path_red}")

print("\n✅ ENTRENAMIENTO DQN AZUL VS ROJO V4 COMPLETADO")
