import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.core.Renderer import Renderer
from HeuristicPolicy import HeuristicPolicy

# Cargar modelo entrenado (PPO rojo)
model = PPO.load("models/ppo_rojo_vs_heuristica_azul.zip")

# Inicializar entorno y heurística
env = StrategyEnvPPOA2C2()
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

print("Presiona ESPACIO para avanzar por cada unidad.")

while not done:
    # Esperar espacio
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

    # Elegir acción
    if current_team == 1:
        action, _ = model.predict(obs, deterministic=True)
        agent_name = "PPO"
    else:
        action = heuristic.get_action(obs)
        agent_name = "Heurística"

    action = int(action)
    action_text = f"{agent_name} ({'rojo' if current_team == 1 else 'azul'})"
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
        active_unit=env.units[env.units.index(unit)] if unit in env.units else None,
        capture_point=env.capture_point,
        capture_score=env.capture_progress,
        max_capture=env.capture_turns_required
    )

pygame.quit()
print("✔ Partida finalizada")
 