import sys
import os
import time
import pygame
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicCastle import HeuristicCastle
from gym_strategy.core.Renderer import Renderer

# Inicializar entorno, heurística y renderer
env = StrategyEnv_Castle(use_obstacles=False)
heuristic_blue = HeuristicCastle(env)
heuristic_red = HeuristicCastle(env)
renderer = Renderer(width=700, height=600, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "↑",
    2: "↓",
    3: "←",
    4: "→"
}

print("🤖 Heurística vs Heurística - Empieza la partida")

while not done:
    team = env.current_player
    team_name = "AZUL" if team == 0 else "ROJO"
    phase = env.phase.upper()
    
    team_units = [u for u in env.units if u.team == team and u.is_alive()]
    unit_index = env.unit_index_per_team[team]
    if unit_index < len(team_units):
        unit = team_units[unit_index]
        ux, uy = unit.position
        print(f"\n🔎 Equipo {team_name} - Fase: {phase}")
        print(f"🧱 Unidad: {unit.unit_type} en ({ux}, {uy}) con {unit.health} HP")

    # Dibujar
    renderer.draw_board(
        units=env.units,
        blocked_positions=[(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]],
        active_unit=unit if unit_index < len(team_units) else None,
        castle_area=env.castle_area,
        castle_hp=env.castle_control
    )

    # Acción
    action = heuristic_blue.get_action(obs) if team == 0 else heuristic_red.get_action(obs)
    print(f"🤖 Acción decidida: {action_names.get(action, str(action))}")

    obs, reward, done, _, _ = env.step(action)
    print(f"✅ Recompensa: {reward:.2f}")

    # Esperar para visualizar
    time.sleep(0.6)

print("\n🎉 Fin de la partida.")
pygame.quit()
