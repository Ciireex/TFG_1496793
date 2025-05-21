import time
import pygame
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.core.Renderer import Renderer

# Wrapper para partida entre dos DQN
class SelfPlayWrapper:
    def __init__(self, env, model_blue, model_red, renderer):
        self.env = env
        self.base_env = env.env
        self.models = {0: model_blue, 1: model_red}
        self.renderer = renderer

    def reset(self):
        obs, _ = self.env.reset()
        self.render_step()
        return obs

    def step(self, obs):
        current_team = self.base_env.current_player
        model = self.models[current_team]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.render_step()
        return obs, reward, terminated or truncated

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

# Cargar modelos
model_blue = DQN.load("models/dqn_vs_heuristic_blue.zip", device="cpu")
model_red = DQN.load("models/dqn_vs_heuristic_red.zip", device="cpu")

# Crear entorno y aplicar flatten
env = StrategyEnvPPOA2C2()
env = FlattenObservation(env)
renderer = Renderer(width=700, height=500, board_size=env.env.board_size)
game = SelfPlayWrapper(env, model_blue, model_red, renderer)

# Iniciar partida
obs = game.reset()
done = False

while not done:
    time.sleep(0.4)
    obs, reward, done = game.step(obs)

print("✅ Partida finalizada.")
time.sleep(2)
pygame.quit()
