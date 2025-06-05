import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv
import gymnasium as gym 

# === Wrapper para entrenar un equipo contra otro modelo PPO ===
class DualTeamVsPPOWrapper(gym.Wrapper):
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
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

# === ENTRENAMIENTO CÍCLICO PPO VS PPO (v3) ===
print("⚔️ Iniciando entrenamiento cíclico PPO Azul vs PPO Rojo (v3)...")

# === Cargar versiones iniciales ===
blue_model = PPO.load("models/ppoblue_vs_heuristic_v2")
red_model = PPO.load("models/pporojo_vs_heuristic_v2")

for ciclo in range(1, 11):
    print(f"\n🌀 CICLO {ciclo} / 10")

    # === Entrenar AZUL contra la versión actual de ROJO ===
    print(f"🎯 Entrenando PPO Azul vs PPO Rojo v3 ciclo {ciclo - 1}")
    env_blue = DummyVecEnv([
        lambda: DualTeamVsPPOWrapper(StrategyEnv(use_obstacles=True), controlled_team=0, opponent_model=red_model)
    ])
    blue_model = PPO(
        "MlpPolicy",
        env_blue,
        verbose=2,
        device="cpu",
        tensorboard_log=f"./logs/ppoblue_vs_pporojo_v3/"
    )
    blue_model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name=f"ciclo_{ciclo}")
    blue_path = f"models/ppoblue_vs_pporojo_v3_ciclo{ciclo}"
    blue_model.save(blue_path)
    print(f"💾 Guardado: {blue_path}")

    # === Entrenar ROJO contra la nueva versión de AZUL ===
    print(f"🎯 Entrenando PPO Rojo vs PPO Azul v3 ciclo {ciclo}")
    env_red = DummyVecEnv([
        lambda: DualTeamVsPPOWrapper(StrategyEnv(use_obstacles=True), controlled_team=1, opponent_model=blue_model)
    ])
    red_model = PPO(
        "MlpPolicy",
        env_red,
        verbose=2,
        device="cpu",
        tensorboard_log=f"./logs/pporojo_vs_ppoblue_v3/"
    )
    red_model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name=f"ciclo_{ciclo}")
    red_path = f"models/pporojo_vs_ppoblue_v3_ciclo{ciclo}"
    red_model.save(red_path)
    print(f"💾 Guardado: {red_path}")

print("\n✅ Entrenamiento cíclico completo.")
