import time
import pygame
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from HeuristicPolicy import HeuristicPolicy
from gym_strategy.core.Renderer import Renderer

class PlayVsHeuristicWrapper:
    def __init__(self, env, controlled_team, renderer):
        self.env = env
        self.base_env = env.env
        self.controlled_team = controlled_team
        self.heuristic = HeuristicPolicy(self.base_env)
        self.renderer = renderer

    def reset(self):
        obs, _ = self.env.reset()
        self.render_step()
        while self.base_env.current_player != self.controlled_team:
            self.render_step()
            time.sleep(0.4)
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                return obs, True
            self.render_step()
        return obs, False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.render_step()
        while not terminated and not truncated and self.base_env.current_player != self.controlled_team:
            time.sleep(0.4)
            self.render_step()
            action = self.heuristic.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.render_step()
        return obs, reward, terminated, truncated, info

    def render_step(self):
        blocked = [(x, y) for x in range(self.base_env.board_size[0])
                          for y in range(self.base_env.board_size[1])
                          if self.base_env.obstacles[x, y]]
        idx = self.base_env.unit_index_per_team[self.base_env.current_player]
        my_units = [u for u in self.base_env.units if u.team == self.base_env.current_player and u.is_alive()]
        unit = my_units[idx] if idx < len(my_units) else None

        self.renderer.draw_board(
            units=self.base_env.units,
            blocked_positions=blocked,
            active_unit=unit,
            capture_point=self.base_env.capture_point,
            capture_score=self.base_env.capture_progress,
            max_capture=self.base_env.capture_turns_required
        )

# Cargar modelo DQN
model = DQN.load("models/dqn_vs_heuristic_blue.zip", device="cpu")

# Crear entorno y renderer
env = StrategyEnvPPOA2C2()
env = FlattenObservation(env)
renderer = Renderer(width=700, height=500, board_size=env.env.board_size)
wrapped_env = PlayVsHeuristicWrapper(env, controlled_team=0, renderer=renderer)

# Iniciar partida
obs, ended = wrapped_env.reset()
done = ended

while not done:
    time.sleep(0.4)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = wrapped_env.step(action)
    done = terminated or truncated

print("✅ Partida finalizada.")
time.sleep(2)
pygame.quit()
