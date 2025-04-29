import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvAdvance2v2CaptureFocus import StrategyEnvAdvance2v2CaptureFocus
from gym_strategy.core.Renderer import Renderer

DIRECTIONS = [
    "quieto", "arriba", "abajo", "izquierda", "derecha",
    "atacar arriba", "atacar abajo", "atacar izquierda", "atacar derecha"
]

# Máscara para PPO
def mask_fn(env):
    return env._get_action_mask()

if __name__ == "__main__":
    # 1. Cargar entorno y modelo
    base_env = StrategyEnvAdvance2v2CaptureFocus()
    env = ActionMasker(base_env, mask_fn)
    model = MaskablePPO.load("ppo_capture_focus")
    renderer = Renderer(width=960, height=400, board_size=(12, 5))

    print("\n¡Comienza la partida de prueba!")

    while True:
        obs, info = env.reset()
        done = False
        turno = 0

        print("\nNuevo mapa generado")

        while not done:
            mask = info["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)

            print(f"\nTurno {turno}:")
            for i, a in enumerate(action):
                unidad = "Soldier" if i == 0 else "Archer"
                print(f"  🧐 {unidad} elige: {DIRECTIONS[a]}")

            obs, reward, done, truncated, info = env.step(action)

            renderer.draw_board(
                units=env.env.units,
                blocked_positions=env.env.blocked_positions,
                capture_point=env.env.capture_point,
                capture_progress=env.env.capture_progress,
                capture_max=env.env.capture_turns_required,
                capturing_team=0
            )

            time.sleep(0.3)
            turno += 1

        # Resultado final
        print(f"\nRecompensa final: {reward:.2f}")
        if reward > 0:
            print("¡Victoria del agente azul!")
        elif reward < 0:
            print("Derrota: la heurística ganó.")
        else:
            print("Empate (timeout)")

        again = input("\n¿Ver otra partida? (s/n): ").lower()
        if again != "s":
            break

    print("Saliendo del juego...")
