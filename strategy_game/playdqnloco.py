import time
import pygame
import numpy as np
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gym_strategy.envs.StrategyEnvPPOA2C import StrategyEnvPPOA2C
from gym_strategy.core.Renderer import Renderer

# Cargar modelos DQN entrenados
blue_model = DQN.load("models/dqn_blue_ppo_a2c.zip")
red_model = DQN.load("models/dqn_red_ppo_a2c.zip")

# Inicializar entorno base y renderer
env = StrategyEnvPPOA2C()
flat_env = FlattenObservation(env)
renderer = Renderer(board_size=env.board_size)
obs, _ = env.reset()

terminated = False
truncated = False
DIRECTIONS = ["quieto", "←", "→", "↑", "↓"]

def predict_action(model, obs):
    return model.predict(obs, deterministic=True)[0]

while not terminated and not truncated:
    print(f"\nTurno del equipo {'AZUL (DQN)' if env.current_player == 0 else 'ROJO (DQN)'} - Fase: {env.phase.upper()}")
    print(f"Progreso de captura - AZUL: {env.capture_progress[0]}/3 | ROJO: {env.capture_progress[1]}/3")

    renderer.draw_board(
        units=env.units,
        blocked_positions=[(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y] == 1],
        capture_point=env.capture_point,
        capture_progress=env.capture_progress,
        capture_max=env.capture_turns_required,
        capturing_team=env.current_player
    )

    flat_obs = flat_env.observation(env._get_obs())
    model = blue_model if env.current_player == 0 else red_model
    action = predict_action(model, flat_obs)
    print("Acción elegida:", DIRECTIONS[action] if action < 5 else action)

    obs, reward, terminated, truncated, info = env.step(action)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

# Resultado final
alive_blue = any(u.is_alive() and u.team == 0 for u in env.units)
alive_red = any(u.is_alive() and u.team == 1 for u in env.units)

print("\n=== RESULTADO FINAL ===")
if env.capture_progress[0] >= env.capture_turns_required:
    print("Gana el EQUIPO AZUL (DQN) por captura")
elif env.capture_progress[1] >= env.capture_turns_required:
    print("Gana el EQUIPO ROJO (DQN) por captura")
elif alive_blue and not alive_red:
    print("Gana el EQUIPO AZUL (DQN) por eliminación")
elif alive_red and not alive_blue:
    print("Gana el EQUIPO ROJO (DQN) por eliminación")
else:
    print("Empate")

print("\nPulsa cualquier tecla para cerrar la ventana...")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            running = False
pygame.quit()
 