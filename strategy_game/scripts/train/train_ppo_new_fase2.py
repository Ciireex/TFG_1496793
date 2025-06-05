import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv_V4 import StrategyEnv_V4
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Wrapper para controlar un equipo y dejar que el otro sea heurístico
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

# === ENTRENAMIENTO PPO AZUL CONTINUACIÓN DESDE v1c ===
print("🏁 Continuando entrenamiento PPO Azul desde ppo_capture_v1c contra heurística...")

env = DummyVecEnv([lambda: DualTeamEnvWrapper(
    StrategyEnv_V4(use_obstacles=True, only_blue=False, enemy_controller="heuristic"),
    controlled_team=0
)])

# Cargar modelo anterior
model = PPO.load("models/ppo_capture_v1c", env=env, device="cpu")

# Continuar entrenamiento con nuevas recompensas y enemigos
model.learn(
    total_timesteps=500_000,
    tb_log_name="ppo_capture_v1c_v2",
    reset_num_timesteps=False
)

model.save("models/ppo_capture_v1c_v2")
print("\n✅ ENTRENAMIENTO COMPLETADO Y GUARDADO COMO models/ppo_capture_v1c_v2 ✅")
