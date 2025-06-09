import os
import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO
import pygame

# === Añadir el path al proyecto ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicDynamicCastle import HeuristicDynamicCastle
from gym_strategy.core.Renderer import Renderer

# === Wrapper PPO vs Heuristic ===
class DualTeamHeuristicWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team=0):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.heuristic = HeuristicDynamicCastle(base_env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # PPO actúa
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Heurística actúa paso a paso
        while self.env.current_player != self.controlled_team:
            team_units = [u for u in self.env.units if u.team == self.env.current_player and u.is_alive()]
            unit = team_units[self.env.unit_index_per_team[self.env.current_player]]
            print(f"🧠 Heurística Roja ({unit.unit_type}) en {unit.position} - Fase: {self.env.phase}")

            h_action = self.heuristic.get_action(obs)
            print(f"👉 Acción heurística: {h_action}\n")
            obs, _, terminated, truncated, info = self.env.step(h_action)

            if terminated or truncated:
                break

        return obs, reward, terminated, truncated, info

# === Cargar entorno y modelo ===
env = DualTeamHeuristicWrapper(StrategyEnv_Castle(use_obstacles=True))
model = PPO.load("models/ppo_castle_vs_azar", env=env, device="cpu")
renderer = Renderer(width=700, height=600, board_size=env.unwrapped.board_size)

# === Jugar una partida ===
obs, _ = env.reset()
done = False
current_attack_dir = (0, 0)
dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

def draw_all():
    team_units = [u for u in env.unwrapped.units if u.team == env.unwrapped.current_player and u.is_alive()]
    index = env.unwrapped.unit_index_per_team[env.unwrapped.current_player]
    active_unit = team_units[index] if index < len(team_units) else None

    highlighted_cells = []
    if active_unit and env.unwrapped.phase == "attack":
        dx, dy = current_attack_dir
        max_range = 3 if active_unit.unit_type == "Archer" else 1
        for dist in range(1, max_range + 1):
            tx, ty = active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist
            if env.unwrapped._valid_coord((tx, ty)):
                highlighted_cells.append((tx, ty))
            else:
                break

    renderer.draw_board(
        units=env.unwrapped.units,
        blocked_positions=[tuple(pos) for pos in zip(*env.unwrapped.obstacles.nonzero())],
        active_unit=active_unit,
        highlight_attack=True,
        castle_area=env.unwrapped.castle_area,
        castle_hp=env.unwrapped.castle_control,
    )

    for x, y in highlighted_cells:
        cw = renderer.width // renderer.board_size[0]
        ch = renderer.height // renderer.board_size[1]
        rect = pygame.Rect(x * cw, y * ch, cw, ch)
        pygame.draw.rect(renderer.screen, (255, 100, 100), rect, 3)

    pygame.display.flip()
    time.sleep(0.4)

# === Bucle de juego ===
while not done:
    draw_all()
    action, _ = model.predict(obs)

    if env.unwrapped.phase == "attack" and action in range(1, 5):
        current_attack_dir = dirs[action]
    else:
        current_attack_dir = (0, 0)

    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print("🎮 Partida terminada.")
