import sys
import os
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sb3_contrib import MaskablePPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy
from gym_strategy.core.Renderer import Renderer

# Cargar modelo MaskablePPO azul v3
model = MaskablePPO.load("models/maskableppoblue_vs_heuristic_v3")

# Iniciar entorno y renderer
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

print("Pulsa ESPACIO para avanzar cada acción. Cierra la ventana para salir.")

while not done:
    pygame.event.pump()

    # Esperar pulsación de espacio
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

    team = env.current_player
    phase = env.phase
    idx = env.unit_index_per_team[team]
    my_units = [u for u in env.units if u.team == team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    if team == 0:
        action, _ = model.predict(obs, deterministic=True)
        agent_name = "MaskablePPO Azul v3"
    else:
        action = heuristic.get_action(obs)
        agent_name = "Heurística Roja v2"

    action = int(action)
    print(f"{agent_name}: {phase.upper()} → {action_names.get(action, '?')}")

    obs, _, done, _, info = env.step(action)

    # Renderizar tablero
    renderer.draw_board(
        units=env.units,
        blocked_positions={(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]},
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("\n🎬 Partida finalizada.")

# Mostrar resultado si existe
if hasattr(env, "winner"):
    reason = getattr(env, "victory_reason", "desconocido")
    if env.winner == 0:
        print("✅ Ganador: MaskablePPO Azul v3", f"(por {reason})")
    elif env.winner == 1:
        print("✅ Ganador: Heurística Roja v2", f"(por {reason})")
    else:
        print("🤝 Empate", f"(por {reason})")
  