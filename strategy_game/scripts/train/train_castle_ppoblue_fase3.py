import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# === Añadir el path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicDynamicCastle import HeuristicDynamicCastle  # ← NUEVA HEURÍSTICA

# === Wrapper PPO vs HeuristicDynamicCastle ===
class DualTeamHeuristicWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team=0):  # PPO = equipo azul
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.heuristic = HeuristicDynamicCastle(base_env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            h_action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, info = self.env.step(h_action)
        return obs, reward, terminated, truncated, info

# === Crear entorno ===
def make_env():
    def _init():
        base_env = StrategyEnv_Castle(use_obstacles=True, obstacle_count=10)
        return DualTeamHeuristicWrapper(base_env, controlled_team=0)
    return _init

env = DummyVecEnv([make_env()])

# === Callback para checkpoints ===
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/",
    name_prefix="ppo_castle_vs_azar"
)

# === Cargar modelo PPO ya entrenado ===
model = PPO.load("models/ppo_castle_vs_dynamicheuristic_continue", env=env, device="cpu")

# === Continuar entrenamiento ===
model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
model.save("models/ppo_castle_vs_azar")
