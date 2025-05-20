
import os
import shutil
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvCNN import StrategyEnvCNN
import gymnasium as gym

class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, env, controlled_team):
        super().__init__(env)
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

def make_dqn_env(team):
    return DummyVecEnv([lambda: DualTeamEnvWrapper(StrategyEnvCNN(), controlled_team=team)])

def evaluate_agent(model, team, n_episodes=5):
    env = DualTeamEnvWrapper(StrategyEnvCNN(), controlled_team=team)
    rewards = []
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
    return np.mean(rewards)

blue_path = "models/dqn_blue_cnn.zip"
red_path = "models/dqn_red_cnn.zip"
backup_dir = "backups_dqn_cnn"
os.makedirs(backup_dir, exist_ok=True)

blue_model = DQN(
    "CnnPolicy",
    make_dqn_env(0),
    verbose=1,
    tensorboard_log="logs/dqn_blue_cnn",
    buffer_size=50000,
    learning_starts=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    device="cpu"
)

red_model = DQN(
    "CnnPolicy",
    make_dqn_env(1),
    verbose=1,
    tensorboard_log="logs/dqn_red_cnn",
    buffer_size=50000,
    learning_starts=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    device="cpu"
)

start_cycle = 1
end_cycle = 10
for cycle in range(start_cycle, end_cycle + 1):
    steps_per_cycle = 10000 if cycle == 1 else 100000

    print(f"\n==== CICLO {cycle} ({steps_per_cycle} pasos) ====")

    print("→ Entrenando equipo AZUL...")
    blue_model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)
    blue_model.save(blue_path)
    shutil.copy(blue_path, os.path.join(backup_dir, f"dqn_blue_cycle{cycle}.zip"))

    mean_rew = evaluate_agent(blue_model, team=0)
    print(f"✔ Azul: recompensa media = {mean_rew:.2f}")

    print("→ Entrenando equipo ROJO...")
    red_model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)
    red_model.save(red_path)
    shutil.copy(red_path, os.path.join(backup_dir, f"dqn_red_cycle{cycle}.zip"))

    mean_rew = evaluate_agent(red_model, team=1)
    print(f"✔ Rojo: recompensa media = {mean_rew:.2f}")

print("\n✅ ENTRENAMIENTO COMPLETADO ✅")
