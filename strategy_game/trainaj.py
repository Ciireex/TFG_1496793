import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvAj import StrategyEnvAj
from small_map_cnn import SmallMapCNN
import gymnasium as gym

class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                break
        print(f"[RESET] Episodio #{self.episode_count + 1} (Equipo {self.controlled_team})")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not (terminated or truncated) and self.env.current_player != self.controlled_team:
            obs, _, terminated, truncated, info = self.env.step(0)
        if terminated or truncated:
            self.episode_count += 1
            print(f"[DONE] Episodio terminado. Recompensa total ≈ {reward:.2f}")
        return obs, reward, terminated, truncated, info

def make_env(team):
    return lambda: DualTeamEnvWrapper(StrategyEnvAj(), controlled_team=team)

def evaluate_agent(model, team, n_episodes=10):
    env = DualTeamEnvWrapper(StrategyEnvAj(), controlled_team=team)
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
        if total >= 2.0:
            victories += 1
    return np.mean(rewards), victories

# Parámetros de red personalizados
policy_kwargs = dict(
    features_extractor_class=SmallMapCNN,
    features_extractor_kwargs={}
)

# Configuración de rutas y modelos
blue_path = "models/ppo_blue_strategy_aj_v3.zip"
red_path = "models/ppo_red_strategy_aj_v3.zip"
backup_dir = "backups_strategy_aj"
os.makedirs(backup_dir, exist_ok=True)

blue_model = PPO(
    "CnnPolicy",
    DummyVecEnv([make_env(0)]),
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="logs/ppo_blue_strategy_aj",
    ent_coef=0.01,
    device="auto"
)

red_model = PPO(
    "CnnPolicy",
    DummyVecEnv([make_env(1)]),
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="logs/ppo_red_strategy_aj",
    ent_coef=0.01,
    device="auto"
)

start_cycle = 1
end_cycle = 20

for cycle in range(start_cycle, end_cycle + 1):
    steps_per_cycle = 100000

    print(f"\n==== CICLO {cycle} ({steps_per_cycle} pasos) ====\n")

    print("→ Entrenando equipo AZUL...")
    blue_model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)
    blue_model.save(blue_path)
    shutil.copy(blue_path, os.path.join(backup_dir, f"ppo_blue_strategy_aj_v3_cycle{cycle}.zip"))

    mean_rew, victories = evaluate_agent(blue_model, team=0, n_episodes=10)
    print(f"✔ Azul: recompensa media = {mean_rew:.2f}, victorias = {victories}/10\n")

    print("→ Entrenando equipo ROJO...")
    red_model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)
    red_model.save(red_path)
    shutil.copy(red_path, os.path.join(backup_dir, f"ppo_red_strategy_aj_v3_cycle{cycle}.zip"))

    mean_rew, victories = evaluate_agent(red_model, team=1, n_episodes=10)
    print(f"✔ Rojo: recompensa media = {mean_rew:.2f}, victorias = {victories}/10\n")

print("\n✅ ENTRENAMIENTO COMPLETADO ✅")
