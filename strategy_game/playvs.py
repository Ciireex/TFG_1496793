import time
import pygame
from stable_baselines3 import PPO, A2C
from gym_strategy.envs.StrategyEnvPPOA2C import StrategyEnvPPOA2C
from gym_strategy.core.Renderer import Renderer

# Cargar modelos entrenados
#blue_model = A2C.load("models/a2c_blue_ppo_a2c_v3.zip")
#red_model = PPO.load("models/ppo_red_ppo_a2c_v3.zip")

blue_model = PPO.load("models/ppo_blue_ppo_a2c_v3.zip")
red_model = A2C.load("models/a2c_red_ppo_a2c_v3.zip")

# Inicializar entorno y renderer
env = StrategyEnvPPOA2C()
renderer = Renderer(board_size=env.board_size)
obs, _ = env.reset()

terminated = False
truncated = False
DIRECTIONS = ["quieto", "←", "→", "↑", "↓"]

def predict_action(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

while not terminated and not truncated:
    print(f"\nTurno del equipo {'AZUL (A2C)' if env.current_player == 0 else 'ROJO (PPO)'} - Fase: {env.phase.upper()}")
    print(f"Progreso de captura - AZUL: {env.capture_progress[0]}/3 | ROJO: {env.capture_progress[1]}/3")

    renderer.draw_board(
        units=env.units,
        blocked_positions=[(x, y) for x in range(env.board_size[0]) for y in range(env.board_size[1]) if env.obstacles[x, y] == 1],
        capture_point=env.capture_point,
        capture_progress=env.capture_progress,
        capture_max=env.capture_turns_required,
        capturing_team=env.current_player
    )

    model = blue_model if env.current_player == 0 else red_model
    action = predict_action(model, obs)
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
    print("Gana el EQUIPO AZUL (A2C) por captura")
elif env.capture_progress[1] >= env.capture_turns_required:
    print("Gana el EQUIPO ROJO (PPO) por captura")
elif alive_blue and not alive_red:
    print("Gana el EQUIPO AZUL (A2C) por eliminación")
elif alive_red and not alive_blue:
    print("Gana el EQUIPO ROJO (PPO) por eliminación")
else:
    print("Empate")

print("\nPulsa cualquier tecla para cerrar la ventana...")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            running = False
pygame.quit() 