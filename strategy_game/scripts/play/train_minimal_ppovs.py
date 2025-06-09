import os
import sys
# === Añadir el path al proyecto ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_Castle_Minimal import StrategyEnv_Castle_Minimal
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# === Wrapper para entrenamiento por turnos con self-play ===
class DualTeamSelfPlayWrapper(gym.Wrapper):
    def __init__(self, env, controlled_team=0, opponent_model=None):
        super().__init__(env)
        self.controlled_team = controlled_team
        self.opponent_model = opponent_model

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            if self.opponent_model:
                action, _ = self.opponent_model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            if self.opponent_model:
                action_op, _ = self.opponent_model.predict(obs, deterministic=True)
            else:
                action_op = self.env.action_space.sample()
            obs, _, terminated, truncated, info = self.env.step(action_op)
        return obs, reward, terminated, truncated, info

# === Crear carpeta de modelos si no existe ===
os.makedirs("models", exist_ok=True)

TOTAL_CYCLES = 10
TIMESTEPS_PER_CYCLE = 100_000

for cycle in range(TOTAL_CYCLES):
    print(f"\n=== CICLO {cycle + 1} ===")

    # === ENTRENAMIENTO AZUL ===
    print("🔵 Entrenando AZUL contra ROJO...")
    red_model = PPO.load("models/ppo_red_latest.zip") if os.path.exists("models/ppo_red_latest.zip") else None

    def make_env_blue():
        base_env = StrategyEnv_Castle_Minimal(use_obstacles=True)
        return DualTeamSelfPlayWrapper(base_env, controlled_team=0, opponent_model=red_model)

    env_blue = DummyVecEnv([make_env_blue])
    model_blue = PPO("MlpPolicy", env_blue, verbose=0)
    model_blue.learn(total_timesteps=TIMESTEPS_PER_CYCLE)
    model_blue.save(f"models/ppo_blue_cycle{cycle + 1}")
    model_blue.save("models/ppo_blue_latest")

    # === ENTRENAMIENTO ROJO ===
    print("🔴 Entrenando ROJO contra AZUL...")
    blue_model = PPO.load("models/ppo_blue_latest.zip")

    def make_env_red():
        base_env = StrategyEnv_Castle_Minimal(use_obstacles=True)
        return DualTeamSelfPlayWrapper(base_env, controlled_team=1, opponent_model=blue_model)

    env_red = DummyVecEnv([make_env_red])
    model_red = PPO("MlpPolicy", env_red, verbose=0)
    model_red.learn(total_timesteps=TIMESTEPS_PER_CYCLE)
    model_red.save(f"models/ppo_red_cycle{cycle + 1}")
    model_red.save("models/ppo_red_latest")

print("✅ Entrenamiento completo.")
