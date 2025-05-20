   
import time
import pygame
from stable_baselines3 import DQN
from gym_strategy.envs.StrategyEnvCNN import StrategyEnvCNN

# Cargar modelos DQN entrenados
blue_model = DQN.load("models/dqn_blue_cnn.zip")
red_model = DQN.load("models/dqn_red_cnn.zip")

# Inicializar entorno
env = StrategyEnvCNN()
obs, _ = env.reset()

terminated = False
truncated = False
DIRECTIONS = ["quieto", "←", "→", "↑", "↓"]

def predict_action(model, obs):
    return model.predict(obs, deterministic=True)[0]

while not terminated and not truncated:
    print(f"\nTurno del equipo {'AZUL (DQN)' if env.current_player == 0 else 'ROJO (DQN)'} - Fase: {env.phase.upper()}")
    print(f"Progreso de captura - AZUL: {env.capture_progress[0]}/3 | ROJO: {env.capture_progress[1]}/3")

    model = blue_model if env.current_player == 0 else red_model
    action = predict_action(model, obs)
    print("Acción elegida:", DIRECTIONS[action] if action < 5 else action)

    obs, reward, terminated, truncated, info = env.step(action)

    # Esperar a que el usuario pulse una tecla o cierre la ventana
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
