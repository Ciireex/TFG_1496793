import sys
import os
import pygame
import numpy as np

# Añadir PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.core.Renderer import Renderer

# Cargar modelo PPO azul entrenado
model = PPO.load("models/ppo_cnn_blue_vs_dummy", device="cpu")

# Dummy policy que siempre pasa
class DummyPolicy:
    def get_action(self, obs):
        return 0

dummy = DummyPolicy()

# Crear entorno y renderer
env = StrategyEnv_Castle(use_obstacles=True)
renderer = Renderer(width=700, height=600, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "izquierda",
    2: "derecha",
    3: "arriba",
    4: "abajo"
}

print("\n🎮 Visualizando PPO Azul vs Dummy Rojo (castillo)")
print("⏩ Pulsa ESPACIO para avanzar cada acción. Cierra la ventana para salir.\n")

while not done:
    pygame.event.pump()

    team_units = [u for u in env.units if u.team == env.current_player and u.is_alive()]
    index = env.unit_index_per_team[env.current_player]
    active_unit = team_units[index] if index < len(team_units) else None

    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0])
                           for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=active_unit,
        terrain=None,
        castle_area=env.castle_area,
        castle_hp=env.castle_control
    )

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

    if env.current_player == 0:
        action, _ = model.predict(obs, deterministic=True)
        print(f"🤖 PPO Azul ({env.phase.upper()} | {active_unit.unit_type}) → {action_names.get(int(action), '?')}")
    else:
        action = dummy.get_action(obs)
        print(f"💤 Dummy Rojo → pasar")

    obs, _, done, _, _ = env.step(int(action))

pygame.quit()
print("\n✅ Partida finalizada.")
