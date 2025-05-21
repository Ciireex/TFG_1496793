import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from HeuristicPolicy import HeuristicPolicy

# Wrapper para controlar un equipo y dejar el otro con heurística
class DualTeamVsHeuristicWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.heuristic = None
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.heuristic is None:
            self.heuristic = HeuristicPolicy(self.env)
        while self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        print(f"[RESET] Episodio #{self.episode_count} (Equipo {self.controlled_team})")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            print(f"[DONE] Episodio terminado. Recompensa total ≈ {reward:.2f}")
        return obs, reward, terminated, truncated, info

# Crear entorno con wrapper y flatten para el equipo rojo
def make_env_vs_heuristic(team):
    return DummyVecEnv([
        lambda: FlattenObservation(DualTeamVsHeuristicWrapper(StrategyEnvPPOA2C2(), controlled_team=team))
    ])

# Evaluación del agente entrenado
def evaluate_agent(model, team, n_episodes=5):
    env = StrategyEnvPPOA2C2()
    env = FlattenObservation(DualTeamVsHeuristicWrapper(env, controlled_team=team))
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total += reward
        print(f"[EVAL] Episodio {i+1}: recompensa total = {total:.2f}")
        rewards.append(total)
    return np.mean(rewards)

# Configuración
model_path = "models/dqn_vs_heuristic_red.zip"
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Entrenamiento para el equipo ROJO (team=1)
env = make_env_vs_heuristic(team=1)

if os.path.exists(model_path):
    model = DQN.load(model_path, env=env, device="cpu")
    print("🔁 Modelo cargado")
else:
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="logs/dqn_vs_heuristic_red",
        device="cpu"
    )
    print("🆕 Nuevo modelo creado")

# Entrenamiento
model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
model.save(model_path)
print("✅ Modelo guardado")

# Evaluación
mean_rew = evaluate_agent(model, team=1)
print(f"\n✔ Recompensa media contra heurística: {mean_rew:.2f}")
