# play_ppored_f0_v3_vs_heuristicblue.py

import os
import sys
import time
import torch
import pygame
import random
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4
from gym_strategy.core.Renderer import Renderer

# === Heurística azul: ataca si puede, si no se mueve aleatoriamente ===
class BlueHeuristicEnv(StrategyEnv_2v2Soldiers4x4):
    def step(self, action):
        if self.current_player == 0:  # Azul (heurística)
            active_unit = self._get_active_unit()
            if not active_unit:
                return super().step(0)

            if self.phase == "attack":
                x, y = active_unit.position
                for dir_idx, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    for dist in range(1, active_unit.attack_range + 1):
                        tx, ty = x + dx * dist, y + dy * dist
                        if not self._valid_coord((tx, ty)):
                            break
                        for enemy in [u for u in self.units if u.team != active_unit.team and u.is_alive()]:
                            if enemy.position == (tx, ty):
                                return super().step(dir_idx + 1)  # ataca
                return super().step(0)  # no enemigo a rango → pasa

            else:  # fase de movimiento
                x, y = active_unit.position
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return super().step(directions.index((dx, dy)) + 1)
                return super().step(0)  # no movimiento válido

        else:
            return super().step(action)  # Rojo (PPO)

def main():
    print("🧠 Cargando modelo PPO rojo...")
    model = PPO.load("./models/ppo_red_vs_heuristicblue/ppo_red_f0_v3")
    print("✅ Modelo PPO cargado correctamente.")

    print("🎮 Inicializando entorno con heurística azul...")
    env = BlueHeuristicEnv()
    renderer = Renderer(width=500, height=500, board_size=env.board_size)

    seed = int(time.time()) % 10000
    obs, _ = env.reset(seed=seed)

    done = False
    clock = pygame.time.Clock()

    while not done:
        if env.current_player == 1:  # Rojo (PPO)
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0  # será ignorado por la heurística

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        renderer.draw_board(
            units=env.units,
            active_unit=env._get_active_unit()
        )

        clock.tick(4)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
