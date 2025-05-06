import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnv2v2 import StrategyEnv2v2
from gym_strategy.core.Renderer import Renderer

# Función para aplicar la máscara de acciones válidas
def mask_fn(env):
    return env._get_action_mask()

DIRECTIONS = [
    "quieto", "↑", "↓", "←", "→",
    "atacar ↑", "atacar ↓", "atacar ←", "atacar →"
]

# Cargar modelo entrenado
model = MaskablePPO.load("ppo_2v2_extremos")

# Crear entorno y renderer
base_env = StrategyEnv2v2()
env = ActionMasker(base_env, mask_fn)
renderer = Renderer(width=600, height=600, board_size=(5, 5))

print("\n🎮 ¡Empieza el duelo 2vs2 contra la IA!")

while True:
    obs, _ = env.reset()
    done = False
    turn = 0

    print("\n🌍 Nuevo mapa generado")

    while not done:
        turn += 1
        current_team = "🔵 Azul" if env.env.current_player == 0 else "🔴 Rojo"
        print(f"\n🎯 TURNO {turn} - Equipo {current_team}")

        mask = obs["action_mask"]
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        print(f"🤖 Acciones elegidas: {[DIRECTIONS[a] for a in action]}")

        obs, reward, done, _, info = env.step(action)

        board_repr = np.full((5, 5), ".")
        for i, unit in enumerate(env.env.units):
            if unit.is_alive():
                symbol = f"A{i}" if unit.team == 0 else f"R{i-2}"
                x, y = unit.position
                board_repr[x, y] = symbol

        cx, cy = env.env.capture_point
        if board_repr[cx, cy] == ".":
            board_repr[cx, cy] = "C"

        print("\n".join(" ".join(row) for row in board_repr))
        print(f"Recompensa: {reward:.2f}")

        renderer.draw_board(
            units=env.env.units,
            blocked_positions=env.env.blocked_positions,
            capture_point=env.env.capture_point,
            capture_progress=env.env.capture_progress[env.env.current_player],
            capture_max=env.env.capture_turns_required,
            capturing_team=env.env.current_player
        )

        time.sleep(0.5)

    print("\n✅ Episodio finalizado")
    if "episode" in info:
        print(f"Resultado: recompensa {info['episode']['r']:.2f}, turnos: {info['episode']['l']}")

    again = input("¿Jugar otra partida? (s/n): ").lower()
    if again != "s":
        break

print("👋 Saliendo del juego...")
