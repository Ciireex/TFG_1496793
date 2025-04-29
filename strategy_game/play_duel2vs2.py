import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvDuel2vs2 import StrategyEnvDuel2vs2
from gym_strategy.core.Renderer import Renderer  # Asegúrate de tener tu Renderer adaptado al tamaño nuevo

# Función para aplicar la máscara de acciones válidas
def mask_fn(env):
    return env._get_action_mask()

DIRECTIONS = [
    "quieto", "arriba", "abajo", "izquierda", "derecha",
    "atacar arriba", "atacar abajo", "atacar izquierda", "atacar derecha"
]

if __name__ == "__main__":
    # 1) Crear el entorno base
    base_env = StrategyEnvDuel2vs2()
    env = ActionMasker(base_env, mask_fn)

    # 2) Cargar el modelo entrenado
    model = MaskablePPO.load("ppo_duel_2v2")

    # 3) Crear el renderer
    renderer = Renderer(width=800, height=600, board_size=(8, 6))  # Adaptado a tablero 8x6

    while True:
        obs, info = env.reset()
        done = False
        current_turn = 0  # Azul empieza

        # 🛠️ Acceso al entorno real para dibujar
        real_env = env.env

        print("\n🌍 Nuevo mapa generado")

        while not done:
            mask = info["action_mask"]

            # 4) Predicción de acciones
            actions, _ = model.predict(obs, deterministic=True, action_masks=mask)

            # 5) Mostrar qué acciones se han elegido
            for i, action in enumerate(actions):
                unit_type = real_env.units[i if current_turn % 2 == 0 else i+2].unit_type
                print(f"Turno {current_turn} | {unit_type} elige: {DIRECTIONS[action]}")

            # 6) Ejecutar acciones
            obs, reward, done, truncated, info = env.step(actions)

            # 7) Dibujar estado
            renderer.draw_board(
                units=real_env.units,
                blocked_positions=real_env.blocked_positions,
                capture_point=real_env.capture_point,
                capture_progress=real_env.capture_progress,
                capture_max=real_env.capture_turns_required,
                capturing_team=current_turn % 2
            )

            time.sleep(0.3)  # Pequeña pausa para ver los movimientos

            current_turn += 1

        # Mostrar resultado
        if reward > 0:
            print("\n🏆 ¡Gana el equipo azul!\n")
        elif reward < 0:
            print("\n🏆 ¡Gana el equipo rojo!\n")
        else:
            print("\n🤝 ¡Empate por agotamiento de turnos!\n")

        # Preguntar si jugar otra partida
        again = input("¿Jugar otro duelo? (s/n): ").lower()
        if again != "s":
            break

    print("👋 Saliendo del juego...")
