import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv

# === Cargar modelo PPO que controlará al equipo rojo ===
opponent_model = PPO.load("models/pporojo_vs_heuristic_curriculum_v1", device="cpu")

# === Wrapper para que A2C controle el equipo azul y el PPO controle el rojo ===
class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team, opponent_model):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.opponent_model = opponent_model
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        print(f"[RESET] Episodio #{self.episode_count}")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

# === ENTRENAMIENTO CON A2C VS PPO ===
print("🏁 Entrenando A2C azul VS PPO rojo...")

env = DummyVecEnv([
    lambda: SelfPlayWrapper(StrategyEnv(use_obstacles=True), controlled_team=0, opponent_model=opponent_model)
])
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/a2cblue_vs_ppored/",
    device="cpu"
)
model.learn(total_timesteps=1_000_000)

# === GUARDAR MODELO ENTRENADO ===
model.save("models/a2cblue_vs_ppored_v1")
print("\n✅ ENTRENAMIENTO COMPLETADO A2C AZUL VS PPO ROJO ✅")
