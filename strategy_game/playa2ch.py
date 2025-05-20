import gymnasium as gym
import pygame
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.core.Renderer import Renderer
from HeuristicPolicy import HeuristicPolicy

# Cargar modelo A2C entrenado
model = A2C.load("models/a2c_vs_heuristic_v2.zip")

# Inicializar entorno y heurística
env = StrategyEnvPPOA2C2()
heuristic = HeuristicPolicy(env)
renderer = Renderer(width=700, height=500, board_size=env.board_size)

obs, _ = env.reset()
done = False

# Diccionario de acciones (coincide con dirs)
action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("Presiona ESPACIO para avanzar por cada unidad.")

while not done:
    # Espera por espacio
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

    current_team = env.current_player
    phase = env.phase

    # Obtener unidad activa
    my_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    idx = env.unit_index_per_team[current_team]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    # Obtener acción
    if current_team == 0:
        action, _ = model.predict(obs, deterministic=True)
        agent_name = "A2C"
    else:
        action = heuristic.get_action(obs)
        agent_name = "Heurística"

    action = int(action)
    action_text = f"{agent_name} ({'azul' if current_team == 0 else 'rojo'})"
    if phase == "move":
        action_text += f" MUEVE {action_names.get(action, '?')}"
    elif phase == "attack":
        action_text += f" ATACA {action_names.get(action, '?')}"
    print(action_text)

    # Ejecutar acción
    obs, reward, done, _, _ = env.step(action)

    # Renderizar
    renderer.draw_board(
        units=env.units,
        blocked_positions={
            (x, y)
            for x in range(env.board_size[0])
            for y in range(env.board_size[1])
            if env.obstacles[x, y]
        },
        active_unit=unit,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
   