import os
import sys
import time
import torch
import pygame
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicCastle import HeuristicCastle
from gym_strategy.core.Renderer import Renderer

class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, env, controlled_team=0, opponent_policy=None):
        super().__init__(env)
        self.controlled_team = controlled_team
        self.opponent_policy = opponent_policy or HeuristicCastle(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            obs, _, terminated, truncated, _ = self.env.step(self.opponent_policy.get_action(obs))
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            obs, reward, terminated, truncated, info = self.env.step(self.opponent_policy.get_action(obs))
        return obs, reward, terminated, truncated, info

# --- Configurar entorno y modelo ---
env = DualTeamEnvWrapper(StrategyEnv_Castle(use_obstacles=True), controlled_team=0)
model = PPO.load("models/ppo_castle_v2_mix", device="cuda" if torch.cuda.is_available() else "cpu")

renderer = Renderer(width=700, height=600, board_size=env.env.board_size)

obs, _ = env.reset()
done = False

clock = pygame.time.Clock()

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    blocked_positions = set(map(tuple, np.argwhere(env.env.obstacles == 1)))
    active_index = env.env.unit_index_per_team[env.env.current_player]
    team_units = [u for u in env.env.units if u.team == env.env.current_player and u.is_alive()]
    active_unit = team_units[active_index] if active_index < len(team_units) else None

    renderer.draw_board(
        units=env.env.units,
        blocked_positions=blocked_positions,
        active_unit=active_unit,
        castle_area=env.env.castle_area,
        castle_hp=env.env.castle_control
    )

    time.sleep(0.8)
    clock.tick(30)
