import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvCaptureMaskedDiscrete import StrategyEnvCaptureMaskedDiscrete
from gym_strategy.core.Renderer import Renderer

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]

def mask_fn(env):
    return env._get_action_mask()

if __name__ == "__main__":
    # 1) Crear el entorno base
    base_env = StrategyEnvCaptureMaskedDiscrete()
    env = ActionMasker(base_env, mask_fn)  # APLICAMOS MASKER TAMBIÉN AQUÍ

    # 2) Cargar el modelo MaskablePPO entrenado
    model = MaskablePPO.load("ppo_capture_masked_v11")

    # 3) Crear el renderer
    renderer = Renderer(width=600, height=600, board_size=(5, 5))

    while True:
        # 4) Resetear el entorno para mapa nuevo
        obs, info = env.reset()
        done = False

        # 🛠️ Corrección para acceder al entorno interno
        real_env = env.env

        print("\n🌍 Nuevo mapa generado!")

        while not done:
            # 5) Obtener máscara de acciones válidas
            mask = info["action_mask"]
            valid = [DIRECTIONS[i] for i, v in enumerate(mask) if v]
            print(f"🎯 Acciones válidas: {valid}")

            # 6) Predicción usando action_masks
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=mask
            )
            action = int(action)
            print(f"🤖 Acción elegida: {DIRECTIONS[action]}")

            # 7) Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)

            # 8) Dibujar estado
            renderer.draw_board(
                units=real_env.units,
                blocked_positions=real_env.blocked_positions,
                capture_point=real_env.capture_point,
                capture_progress=real_env.capture_progress,
                capture_max=real_env.capture_turns_required,
                capturing_team=0
            )

            time.sleep(0.3)

        print(f"🏁 Partida terminada - Recompensa final: {reward:.2f}")

        # Preguntar si quieres jugar otra
        again = input("¿Quieres jugar otro mapa? (s/n): ").lower()
        if again != "s":
            break

    print("👋 Saliendo del juego...")
