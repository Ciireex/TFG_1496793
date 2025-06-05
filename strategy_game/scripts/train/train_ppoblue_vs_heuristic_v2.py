import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
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

# === ENTRENAMIENTO COMPLETO CON OBSTÁCULOS ===
print("🏁 Entrenando PPO vs heurística (con obstáculos, 1M pasos)...")
env = DummyVecEnv([
    lambda: DualTeamEnvWrapper(StrategyEnv(use_obstacles=True), controlled_team=0)
])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppoblue_vs_heuristic/",
    device="cpu"
)

model.learn(
    total_timesteps=1_000_000,
    tb_log_name="PPO_vs_Heuristic",  # 📌 Nombre único para ver toda la curva completa
    reset_num_timesteps=True         # 🔁 Muestra correctamente desde t=0 hasta 1M
)

# Guardar modelo final
model.save("models/ppoblue_vs_heuristic_v1")

print("\n✅ ENTRENAMIENTO COMPLETADO Y GUARDADO ✅")
