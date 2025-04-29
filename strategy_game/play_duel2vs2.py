import time
import numpy as np
import pygame
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvAdvance2v2 import StrategyEnvAdvance2v2
from gym_strategy.core.Renderer import Renderer

DIRECTIONS = [
    "quieto", "arriba", "abajo", "izquierda", "derecha",
    "atacar arriba", "atacar abajo", "atacar izquierda", "atacar derecha"
]

# Función para aplicar la máscara
def mask_fn(env):
    return env._get_action_mask()

if __name__ == "__main__":
    # 1) Crear entorno
    base_env = StrategyEnvAdvance2v2()
    env = ActionMasker(base_env, mask_fn)

    # 2) Cargar modelo
    model = MaskablePPO.load("ppo_advance2v2_final")

    # 3) Crear renderer
    renderer = Renderer(width=960, height=400, board_size=(12, 5))

    print("\n🎮 ¡Empieza la partida estilo Advance Wars!")

    while True:
        obs, info = env.reset()
        done = False
        current_team = 0  # 0: azul, 1: rojo
        turn = 0

        print("\n🌍 Nuevo mapa generado")

        while not done:
            masks = info["action_mask"]

            # 4) Predecir acción de ambas unidades
            actions, _ = model.predict(obs, deterministic=True, action_masks=masks)
            actions = np.array(actions)

            # Mostrar acciones
            for idx, action in enumerate(actions):
                unit_type = "Soldier" if idx == 0 else "Archer"
                player_color = "🔵 Azul" if current_team == 0 else "🔴 Rojo"
                print(f"Turno {turn} | {player_color} - {unit_type} elige: {DIRECTIONS[action]}")

            # 5) Ejecutar acciones
            obs, reward, done, truncated, info = env.step(actions)

            # 6) Dibujar estado
            real_env = env.env  # Acceso interno al entorno sin ActionMasker
            renderer.draw_board(
                units=real_env.units,
                blocked_positions=real_env.blocked_positions,
                capture_point=real_env.capture_point,
                capture_progress=real_env.capture_progress,
                capture_max=real_env.capture_turns_required,
                capturing_team=current_team
            )

            if not done:
                current_team = 1 - current_team  # Cambiar equipo
                turn += 1

            time.sleep(0.4)  # 🔵 Espera para ver las jugadas

        # Resultado
        if reward > 0:
            winner = "🔵 Azul" if current_team == 0 else "🔴 Rojo"
            print(f"\n🏆 ¡Gana el equipo {winner}!\n")
        else:
            print("\n🤝 ¡Empate por agotamiento de turnos!\n")

        again = input("¿Jugar otra partida? (s/n): ").lower()
        if again != "s":
            break

    print("👋 Saliendo del juego...")
