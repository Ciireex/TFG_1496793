import sys
import os
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy
from gym_strategy.core.Renderer import Renderer

# Inicializar entorno y render
env = StrategyEnv(use_obstacles=True)
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("Controles:")
print("0: pasar | 1: ← | 2: → | 3: ↑ | 4: ↓")

while not done:
    pygame.event.pump()

    current_team = env.current_player
    phase = env.phase
    idx = env.unit_index_per_team[current_team]
    my_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    if current_team == 0:
        # Input humano
        valid = False
        while not valid:
            try:
                action = int(input(f"TURNO HUMANO ({phase.upper()}), unidad {unit.unit_type} en {unit.position}: "))
                if action in [0, 1, 2, 3, 4]:
                    valid = True
            except ValueError:
                pass
        print(f"👤 HUMANO: {phase.upper()} {action_names.get(action)}")
    else:
        # Heurística
        action = heuristic.get_action(obs)
        print(f"🤖 HEURÍSTICA: {phase.upper()} {action_names.get(action)}")

    obs, reward, done, _, _ = env.step(action)

    # Renderizar
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
