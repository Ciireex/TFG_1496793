import sys, os
sys.path.append(os.path.abspath("."))  # Asegura que se pueda importar gym_strategy

import time
import numpy as np
import pygame
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import A2C

from gym_strategy.envs.Env3v3 import Env3v3
from gym_strategy.core.Unit import Soldier, Archer
from gym_strategy.core.Renderer import Renderer

DIRECTIONS = [
    "quieto", "↑", "↓", "←", "→",
    "atacar ↑", "atacar ↓", "atacar ←", "atacar →"
]

# Cambia el modelo aquí
MODEL_NAME = "ppo_3v3_soldiers_archers"  # o "a2c_3v3_soldiers_archers"
USE_MASKABLE = MODEL_NAME.startswith("ppo")

def mask_fn(env):
    return env._get_action_mask()

# Equipos
blue_team = [Soldier, Soldier, Archer]
red_team = [Archer, Soldier, Soldier]

# Cargar modelo
if USE_MASKABLE:
    model = MaskablePPO.load(MODEL_NAME)
else:
    model = A2C.load(MODEL_NAME)

# Crear entorno
base_env = Env3v3(blue_team=blue_team, red_team=red_team)
env = ActionMasker(base_env, mask_fn) if USE_MASKABLE else base_env

# Crear renderer visual
renderer = Renderer(width=700, height=700, board_size=(7, 7))

print("\n¡Empieza el duelo 3vs3 contra la IA!")

while True:
    obs, _ = env.reset()
    done = False
    turn = 0
    print("\nNuevo mapa generado")

    while not done:
        turn += 1
        current_team = "Azul" if env.env.current_player == 0 else "Rojo"
        print(f"\nTURNO {turn} - Equipo {current_team}")

        mask = obs["action_mask"] if USE_MASKABLE else None
        action, _ = model.predict(obs, deterministic=True, action_masks=mask) if USE_MASKABLE else model.predict(obs, deterministic=True)
        print(f"Acciones elegidas: {[DIRECTIONS[a] for a in action]}")

        obs, reward, done, _, info = env.step(action)

        renderer.draw_board(
            units=env.env.units,
            blocked_positions=env.env.blocked_positions,
            capture_point=env.env.capture_point,
            capture_progress=env.env.capture_progress,
            capture_max=env.env.capture_turns_required,
            capturing_team=env.env.current_player
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        time.sleep(0.5)

    print("\nEpisodio finalizado")
    if "episode" in info:
        print(f"Resultado: recompensa {info['episode']['r']:.2f}, turnos: {info['episode']['l']}")

    again = input("¿Jugar otra partida? (s/n): ").lower()
    if again != "s":
        break

print("Saliendo del juego...")
pygame.quit()
