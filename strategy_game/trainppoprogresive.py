import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvPvP import StrategyEnvPvP
from CustomCNN import SmallCNN

# --- Entrenamiento PPO contra heurística básica ---

def heuristic_action(env, unit):
    def distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Atacar si puede
    for dir_idx, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)], start=1):
        for dist in range(1, 4 if unit.unit_type == "Archer" else 2):
            tx = unit.position[0] + dx * dist
            ty = unit.position[1] + dy * dist
            if not (0 <= tx < env.board_size[0] and 0 <= ty < env.board_size[1]):
                break
            for target in env.units:
                if target.team != unit.team and target.position == (tx, ty) and target.is_alive():
                    return dir_idx
            if unit.unit_type != "Archer":
                break

    # Mover hacia enemigo más cercano
    enemies = [u for u in env.units if u.team != unit.team and u.is_alive()]
    if not enemies:
        return 0
    target = min(enemies, key=lambda e: distance(unit.position, e.position))
    ux, uy = unit.position
    tx, ty = target.position
    if abs(tx - ux) > abs(ty - uy):
        return 2 if tx > ux else 4
    else:
        return 3 if ty > uy else 1

class HeuristicWrapper(gym.Wrapper):
    def __init__(self, env, controlled_team=0):
        super().__init__(env)
        self.controlled_team = controlled_team
        self.current_episode_reward = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            unit = self.env.get_active_unit()
            action = heuristic_action(self.env, unit) if unit else 0
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.current_episode_reward = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_reward += reward
        while not (terminated or truncated) and self.env.current_player != self.controlled_team:
            unit = self.env.get_active_unit()
            action = heuristic_action(self.env, unit) if unit else 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_episode_reward += reward
        if terminated or truncated:
            print(f"[DONE] Recompensa total ≈ {self.current_episode_reward:.2f}")
        return obs, reward, terminated, truncated, info

def main():
    model_path = "models/ppo_turnbased_vs_heuristic.zip"
    os.makedirs("models", exist_ok=True)

    def make_env():
        base_env = StrategyEnvPvP()
        return HeuristicWrapper(base_env, controlled_team=0)

    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="logs_ppo_turnbased",
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=500_000)
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()
