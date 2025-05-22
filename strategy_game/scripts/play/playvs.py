import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.core.Renderer import Renderer

# Cargar modelo PPO azul
model_blue = PPO.load("models/ppoblue_v2")

# Inicializar entorno y renderer
env = StrategyEnv(use_obstacles=True)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Mapeo de teclas del humano
key_to_action = {
    pygame.K_0: 0,  # pasar
    pygame.K_1: 1,  # izquierda
    pygame.K_2: 2,  # derecha
    pygame.K_3: 3,  # arriba
    pygame.K_4: 4   # abajo
}

action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("🎮 CONTROL HUMANO (equipo ROJO)")
print("Pulsa: 1=←, 2=→, 3=↑, 4=↓, 0=pasar")
print("Pulsa ESPACIO para que la IA (azul) juegue su turno.")

while not done:
    current_team = env.current_player
    phase = env.phase
    idx = env.unit_index_per_team[current_team]
    my_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    # IA (azul)
    if current_team == 0:
        print(f"\n🤖 Turno IA AZUL ({phase.upper()}) — Pulsa ESPACIO para ver su acción...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        action, _ = model_blue.predict(obs, deterministic=True)
        print(f"🤖 PPO (azul): {phase.upper()} {action_names.get(int(action), '?')}")

    # Humano (rojo)
    else:
        print(f"\n🧑‍💻 Tu turno ROJO ({phase.upper()}) — Pulsa tecla 1=←, 2=→, 3=↑, 4=↓, 0=pasar")
        action = None
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in key_to_action:
                    action = key_to_action[event.key]
                    waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        print(f"🧑‍💻 Humano: {phase.upper()} {action_names.get(int(action), '?')}")

    # Ejecutar acción
    obs, reward, done, _, _ = env.step(action)

    # Dibujar tablero
    renderer.draw_board(
        units=env.units,
        blocked_positions={
            (x, y) for x in range(env.board_size[0])
            for y in range(env.board_size[1]) if env.obstacles[x, y]
        },
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
