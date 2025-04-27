import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvDuel import StrategyEnvDuel
from gym_strategy.core.Renderer import Renderer

# Función para aplicar máscara
def mask_fn(env):
    return env._get_action_mask()

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha", "atacar arriba", "atacar abajo", "atacar izquierda", "atacar derecha"]

if __name__ == "__main__":
    # 1) Crear el entorno
    base_env = StrategyEnvDuel()
    env = ActionMasker(base_env, mask_fn)

    # 2) Cargar el modelo
    model = MaskablePPO.load("ppo_duel_v1")

    # 3) Crear el renderer
    renderer = Renderer(width=600, height=600, board_size=(5, 5))

    print("\n🎮 ¡Empieza el duelo!")

    while True:
        obs, info = env.reset()
        done = False
        current_player = 0
        turn = 0

        real_env = env.env  # 🛠️ Acceso al entorno real para el renderer

        print("\n🌍 Nuevo mapa generado")

        while not done:
            mask = info["action_mask"]

            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)

            player_color = "🔵 Azul" if current_player == 0 else "🔴 Rojo"
            print(f"Turno {turn} | {player_color} elige acción: {DIRECTIONS[action]}")

            obs, reward, done, truncated, info = env.step(action)

            # 🖼️ Dibujar el tablero actualizado
            renderer.draw_board(
                units=real_env.units,
                blocked_positions=real_env.blocked_positions,
                capture_point=real_env.capture_point,
                capture_progress=real_env.capture_progress[current_player],
                capture_max=real_env.capture_turns_required,
                capturing_team=current_player
            )

            if not done:
                current_player = 1 - current_player
                turn += 1

            time.sleep(0.3)  # 💤 Pequeña espera para que puedas ver el render

        # Mostrar resultado
        if reward > 0:
            winner = "🔵 Azul" if current_player == 0 else "🔴 Rojo"
            print(f"\n🏆 ¡Gana el equipo {winner}!\n")
        else:
            print("\n🤝 ¡Empate por agotamiento de turnos!\n")

        again = input("¿Jugar otro duelo? (s/n): ").lower()
        if again != "s":
            break

    print("👋 Saliendo del juego...")
