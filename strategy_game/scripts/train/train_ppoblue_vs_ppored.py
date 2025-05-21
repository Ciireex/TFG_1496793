import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Wrapper para entrenar un equipo y dejar que el otro lo controle un PPO congelado
class DualTeamSelfPlayWrapper(gym.Wrapper):
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

# === Configuración ===
total_cycles = 10
timesteps_per_cycle = 100_000

# === Cargar modelos iniciales base ===
print("📥 Cargando modelos base...")
model_blue = PPO.load("models/ppoblue_vs_heuristic_curriculum_v1")
model_red = PPO.load("models/pporojo_vs_heuristic_curriculum_v1")

# === Entrenamiento alterno ===
for cycle in range(total_cycles):
    print(f"\n🔁 CICLO {cycle + 1}/{total_cycles} - Entrena AZUL contra ROJO_v2")
    # Cargar la versión más reciente del rojo si existe
    if os.path.exists("models/ppored_v2.zip"):
        opponent_red = PPO.load("models/ppored_v2")
    else:
        opponent_red = model_red
    env_train_blue = DummyVecEnv([
        lambda: DualTeamSelfPlayWrapper(StrategyEnv(use_obstacles=True), controlled_team=0, opponent_model=opponent_red)
    ])
    model_blue.set_env(env_train_blue)
    model_blue.learn(total_timesteps=timesteps_per_cycle)
    model_blue.save("models/ppoblue_v2")

    print(f"🔁 CICLO {cycle + 1}/{total_cycles} - Entrena ROJO contra AZUL_v2")
    if os.path.exists("models/ppoblue_v2.zip"):
        opponent_blue = PPO.load("models/ppoblue_v2")
    else:
        opponent_blue = model_blue
    env_train_red = DummyVecEnv([
        lambda: DualTeamSelfPlayWrapper(StrategyEnv(use_obstacles=True), controlled_team=1, opponent_model=opponent_blue)
    ])
    model_red.set_env(env_train_red)
    model_red.learn(total_timesteps=timesteps_per_cycle)
    model_red.save("models/ppored_v2")

print("\n✅ ENTRENAMIENTO COMPLETADO - PPO AZUL VS PPO ROJO (v2) ✅")
