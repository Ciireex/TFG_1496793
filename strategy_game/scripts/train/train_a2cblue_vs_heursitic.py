import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Wrapper para que un equipo juegue y el otro sea heurístico
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

# === FASE 1: Sin obstáculos ===
print("🏁 Entrenando sin obstáculos (A2C)...")
env_no_obs = DummyVecEnv([
    lambda: DualTeamEnvWrapper(StrategyEnv(use_obstacles=False), controlled_team=0)
])
model = A2C(
    "MlpPolicy",
    env_no_obs,
    verbose=1,
    tensorboard_log="./logs/a2cblue_vs_heuristic_curriculum_v2/",
    device="cpu"
)
model.learn(total_timesteps=1_000_000)

# === FASE 2: Con obstáculos ===
print("🏁 Entrenando con obstáculos (A2C)...")
env_with_obs = DummyVecEnv([
    lambda: DualTeamEnvWrapper(StrategyEnv(use_obstacles=True), controlled_team=0)
])
model.set_env(env_with_obs)
model.learn(total_timesteps=1_000_000)

# Guardar modelo final
model.save("models/a2cblue_vs_heuristic_curriculum_v2")

print("\n✅ ENTRENAMIENTO COMPLETADO A2C AZUL VS HEURÍSTICA (v2) ✅")
