import os
import sys
import time
import pygame
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicCastle import HeuristicCastle
from gym_strategy.core.Renderer import Renderer

# === Wrapper para visualizar PPO vs HeuristicCastle ===
class DualTeamHeuristicWrapper:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.heuristic = HeuristicCastle(env)
        self.current_obs, _ = env.reset()

    def play_step(self):
        if self.env.current_player == 0:
            action, _ = self.model.predict(self.current_obs, deterministic=True)
        else:
            action = self.heuristic.get_action(self.current_obs)

        self.current_obs, _, terminated, truncated, _ = self.env.step(action)
        return terminated or truncated

# === Cargar modelo PPO entrenado ===
model = PPO.load("models/ppo_castle_vs_heuristic_continue", device="cpu")

# === Crear entorno base y renderer ===
base_env = StrategyEnv_Castle(use_obstacles=True, obstacle_count=10)
renderer = Renderer(width=700, height=600, board_size=base_env.board_size)
wrapped_env = DualTeamHeuristicWrapper(base_env, model)

done = False

while not done:
    renderer.draw_board(
        units=base_env.units,
        blocked_positions=list(zip(*base_env.obstacles.nonzero())),
        active_unit=base_env.units[base_env.unit_index_per_team[base_env.current_player]]
        if base_env.unit_index_per_team[base_env.current_player] < len(base_env.units) else None,
        castle_area=base_env.castle_area,
        castle_hp=base_env.castle_control
    )

    time.sleep(0.2)
    done = wrapped_env.play_step()

pygame.quit()
