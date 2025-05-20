import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from HeuristicPolicy import HeuristicPolicy
import gymnasium as gym

# --- Wrapper Heurística en azul (equipo 0), entrenar PPO en rojo (equipo 1) ---
class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team=1):
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

# --- Crear entorno ---
def make_env():
    return lambda: DualTeamEnvWrapper(StrategyEnvPPOA2C2(), controlled_team=1)

env = DummyVecEnv([make_env()])
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo_rojo_vs_heuristica_azul", device="cpu")

# --- Entrenamiento ---
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps)
model.save("models/ppo_rojo_vs_heuristica_azul")

print("\n✅ ENTRENAMIENTO COMPLETADO (PPO rojo vs Heurística azul) ✅")
