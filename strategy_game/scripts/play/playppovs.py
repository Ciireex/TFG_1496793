import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
blue_model = PPO.load("models/selfplay_ppo_blue_cycle10.zip", device="cpu")
red_model = PPO.load("models/selfplay_ppo_red_cycle10.zip", device="cpu")

# Inicializar entorno
env = StrategyEnvPPOA2C2()
renderer = Renderer(width=700, height=500, board_size=env.board_size)
obs, _ = env.reset()
done = False

# Diccionario de acciones (coincide con dirs del entorno)
action_names = {
    0: "pasar",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}

print("Presiona ESPACIO para avanzar por cada unidad.")

while not done:
    # Espera a pulsar espacio
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
    idx = env.unit_index_per_team[current_team]
    my_units = [u for u in env.units if u.team == current_team and u.is_alive()]
    unit = my_units[idx] if idx < len(my_units) else None

    if unit is None or not unit.is_alive():
        obs, _, done, _, _ = env.step(0)
        continue

    if current_team == 0:
        action, _ = blue_model.predict(obs, deterministic=True)
        agent = "AZUL"
    else:
        action, _ = red_model.predict(obs, deterministic=True)
        agent = "ROJO"

    action = int(action)
    action_text = f"{agent} {'MUEVE' if phase == 'move' else 'ATACA'} {action_names.get(action, '?')}"
    print(action_text)

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
