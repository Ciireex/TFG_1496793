import pygame
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.core.Renderer import Renderer

# Acción por dirección según tu codificación
DIRECTION_TO_ACTION = {
    "stay": 0,   # pasar
    "left": 1,   # ←
    "right": 2,  # →
    "up": 3,     # ↑
    "down": 4    # ↓
}

# Teclas físicas → direcciones fijas
KEY_TO_ACTION = {
    pygame.K_q: DIRECTION_TO_ACTION["stay"],
    pygame.K_a: DIRECTION_TO_ACTION["left"],
    pygame.K_d: DIRECTION_TO_ACTION["right"],
    pygame.K_w: DIRECTION_TO_ACTION["up"],
    pygame.K_s: DIRECTION_TO_ACTION["down"],
}

# Inicializar entorno y renderer
env = StrategyEnv_Castle()
renderer = Renderer(width=700, height=600, board_size=env.board_size)

obs, _ = env.reset()
done = False
pygame.init()

print("🎮 Controles (idénticos para ambos equipos):")
print("    W = arriba, S = abajo, A = izquierda, D = derecha, Q = pasar")

# Bucle principal
while not done:
    # Obtener unidad activa
    team_units = [u for u in env.units if u.team == env.current_player and u.is_alive()]
    if env.unit_index_per_team[env.current_player] < len(team_units):
        active_unit = team_units[env.unit_index_per_team[env.current_player]]
    else:
        active_unit = None

    renderer.draw_board(
        units=env.units,
        blocked_positions=[(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y]],
        active_unit=active_unit,
        castle_area=env.castle_area,
        castle_hp=env.castle_control
    )

    print(f"\n🔎 Equipo {'AZUL' if env.current_player == 0 else 'ROJO'} - Fase: {env.phase.upper()}")
    if active_unit:
        print(f"🧱 Unidad activa: {active_unit.unit_type} en {active_unit.position} con {active_unit.health} HP")

    # Esperar a pulsación válida
    waiting = True
    action = 0
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in KEY_TO_ACTION:
                action = KEY_TO_ACTION[event.key]
                waiting = False

    # Ejecutar acción
    obs, reward, done, _, _ = env.step(action)
    print(f"✅ Acción ejecutada: {action} | Recompensa: {reward:.2f}")

print("🎉 Partida finalizada.")
pygame.quit()
