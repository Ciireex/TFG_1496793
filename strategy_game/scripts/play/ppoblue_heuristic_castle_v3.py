import os
import sys
import time
import torch
import pygame
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.utils.HeuristicCastle import HeuristicCastle
from gym_strategy.core.Renderer import Renderer

# === Wrapper para IA PPO contra heurística ===
class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, env, model_path, controlled_team=0):
        super().__init__(env)
        self.controlled_team = controlled_team
        self.model = PPO.load(model_path, device="cpu",
                              custom_objects={"features_extractor_class": CustomCNN})
        self.heuristic = HeuristicCastle(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, obs):
        team = int(obs[18, 0, 0])  # 0 = azul, 1 = rojo

        if team == self.controlled_team:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            action = self.heuristic.get_action(obs)

        obs, reward, terminated, truncated, info = self.env.step(action)
        print(f"🔁 Acción: {action}, Recompensa: {reward:.2f}")
        return obs, reward, terminated, truncated, info

    def render(self, renderer):
        blocked = np.argwhere(self.env.obstacles == 1)
        blocked_list = [tuple(pos) for pos in blocked]

        renderer.draw_board(
            units=self.env.units,
            blocked_positions=blocked_list,
            active_unit=self._get_active_unit(),
            castle_area=self.env.castle_area,
            castle_hp=self.env.castle_control
        )

    def _get_active_unit(self):
        team_units = [u for u in self.env.units if u.team == self.env.current_player and u.is_alive()]
        idx = self.env.unit_index_per_team[self.env.current_player]
        if idx < len(team_units):
            return team_units[idx]
        return None

# === Main ===
if __name__ == "__main__":
    env = StrategyEnv_Castle(use_obstacles=True, obstacle_count=10)
    wrapper = DualTeamEnvWrapper(env, model_path="models/ppo_castle_solo_v1", controlled_team=0)
    renderer = Renderer(width=700, height=600, board_size=env.board_size)

    obs, _ = wrapper.reset()
    done = False

    while not done:
        obs, reward, terminated, truncated, _ = wrapper.step(obs)
        done = terminated or truncated
        wrapper.render(renderer)
        time.sleep(0.5)
