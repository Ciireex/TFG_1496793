import pygame
import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvAj import StrategyEnvAj
from gym_strategy.core.Renderer import Renderer
from gym_strategy.core.Unit import Soldier, Archer, Knight

# Cargar modelos entrenados versión v3
model_blue = PPO.load("models/ppo_blue_strategy_aj_v3.zip")
model_red = PPO.load("models/ppo_red_strategy_aj_v3.zip")

# Inicializar entorno y renderer
env = StrategyEnvAj()
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Diccionario corregido según el renderizado en pantalla
acciones = {
    0: "quieto",
    1: "← (izquierda)",
    2: "→ (derecha)",
    3: "↑ (arriba)",
    4: "↓ (abajo)"
}

while not done:
    current_team = env.current_player
    active_unit = env.units[env.active_unit_index] if env.active_unit_index < len(env.units) else None

    model = model_blue if current_team == 0 else model_red
    action, _ = model.predict(obs, deterministic=True)

    if active_unit:
        fase = env.phase.upper()
        action_int = int(action)
        print(f"[{fase}] Equipo {'AZUL' if current_team == 0 else 'ROJO'} | {active_unit.unit_type} en {active_unit.position} → Acción: {action_int} ({acciones[action_int]})")

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    renderer.draw_board(
        units=env.units,
        blocked_positions={
            (x, y)
            for x in range(env.board_size[0])
            for y in range(env.board_size[1])
            if env.obstacles[x, y] == 1
        },
        capture_point=env.capture_point,
        capture_progress=env.capture_progress,
        capture_max=env.capture_turns_required,
        active_unit=active_unit,
        highlight_attack=True
    )

    # Esperar a que se pulse ESPACIO para continuar
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

# Fin de partida
print("Partida finalizada. Pulsa una tecla para salir.")
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            pygame.quit()
            exit()
