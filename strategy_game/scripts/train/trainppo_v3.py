import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
import gymnasium as gym

class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        print(f"[RESET] Episodio #{self.episode_count + 1} (Equipo {self.controlled_team})")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.episode_count += 1
            print(f"[DONE] Episodio terminado. Recompensa total ≈ {reward:.2f}")
        return obs, reward, terminated, truncated, info

def make_env(team):
    return lambda: DualTeamEnvWrapper(StrategyEnvPPOA2C2(), controlled_team=team)

def evaluate_agent(model, team, n_episodes=5):
    env = DualTeamEnvWrapper(StrategyEnvPPOA2C2(), controlled_team=team)
    rewards = []
    victories = 0
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"[EVAL] Episodio {i+1}: recompensa total = {total:.2f}")
        rewards.append(total)
        if total >= 3.0:
            victories += 1
    return np.mean(rewards), victories

# Rutas
blue_path = "models/ppoblue_capture.zip"
red_path = "models/ppored_capture.zip"
backup_dir = "backups_capture"
os.makedirs(backup_dir, exist_ok=True)

# Modelos
blue_model = PPO("MlpPolicy", DummyVecEnv([make_env(0)]), verbose=1, tensorboard_log="logs/ppoblue_capture", device="cpu")
red_model = PPO("MlpPolicy", DummyVecEnv([make_env(1)]), verbose=1, tensorboard_log="logs/ppored_capture", device="cpu")

# Entrenamiento
start_cycle = 1
end_cycle = 5

for cycle in range(start_cycle, end_cycle + 1):
    print(f"\n==== CICLO {cycle} ====")

    print("→ Entrenando BLUE contra RED actual...")
    blue_model.set_env(DummyVecEnv([make_env(0)]))
    blue_model.learn(total_timesteps=100000, reset_num_timesteps=False)
    blue_model.save(blue_path)
    shutil.copy(blue_path, os.path.join(backup_dir, f"ppoblue_capture_cycle{cycle}.zip"))
    mean_rew, victories = evaluate_agent(blue_model, team=0)
    print(f"✔ Blue: recompensa media = {mean_rew:.2f}, victorias = {victories}/5\n")

    print("→ Entrenando RED contra BLUE actual...")
    red_model.set_env(DummyVecEnv([make_env(1)]))
    red_model.learn(total_timesteps=100000, reset_num_timesteps=False)
    red_model.save(red_path)
    shutil.copy(red_path, os.path.join(backup_dir, f"ppored_capture_cycle{cycle}.zip"))
    mean_rew, victories = evaluate_agent(red_model, team=1)
    print(f"✔ Red: recompensa media = {mean_rew:.2f}, victorias = {victories}/5\n")

print("\n✅ ENTRENAMIENTO COMPLETADO ✅")
