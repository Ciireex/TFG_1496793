import os
import sys
import time
import pygame
from stable_baselines3 import PPO

# Añadir la ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase1_Soldiers4x4
from gym_strategy.core.Renderer import Renderer

# === ENVOLTORIO: modelo azul vs heurística roja ===
class EnvBlueVsHeuristic(Env_Fase1_Soldiers4x4):
    def __init__(self, model_blue):
        super().__init__()
        self.model_blue = model_blue

    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:
            action, _ = self.model_blue.predict(obs, deterministic=True)
            return super().step(action)
        else:
            unit = self._get_active_unit()
            if not unit:
                return super().step(0)

            if self.phase == "attack":
                x, y = unit.position
                for dir_idx, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    tx, ty = x + dx, y + dy
                    if not self._valid_coord((tx, ty)):
                        continue
                    for enemy in [u for u in self.units if u.team != unit.team and u.is_alive()]:
                        if enemy.position == (tx, ty):
                            return super().step(dir_idx + 1)
                return super().step(0)
            else:
                x, y = unit.position
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return super().step(directions.index((dx, dy)) + 1)
                return super().step(0)

def main():
    CURRENT_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_blue_def_f0.zip"))

    print("🧠 Cargando modelo PPO azul...")
    model_blue = PPO.load(MODEL_PATH)
    print("✅ Modelo cargado.")

    print("🎮 Inicializando entorno (4x4, 2 soldados por bando)...")
    env = EnvBlueVsHeuristic(model_blue)
    renderer = Renderer(width=400, height=400, board_size=env.board_size)

    seed = int(time.time() * 1000) % 2**32
    obs, _ = env.reset(seed=seed)
    print(f"🌱 Semilla de partida: {seed}")

    done = False
    clock = pygame.time.Clock()

    while not done:
        obs, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            active_unit=env._get_active_unit(),
            blocked_positions=(env.terrain == 99)
        )

        clock.tick(4)  # velocidad de la simulación

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
