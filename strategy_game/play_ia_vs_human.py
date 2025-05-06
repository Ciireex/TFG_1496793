import numpy as np
from gym_strategy.envs.StrategyEnv2v2 import StrategyEnv2v2
from sb3_contrib.ppo_mask import MaskablePPO

# Cargar modelo entrenado
model = MaskablePPO.load("ppo_2v2_v1")

# Crear entorno
env = StrategyEnv2v2()
obs, _ = env.reset()
done = False
turn = 0

print("\n🎮 Controles: ingresa un número del 0 al 8 para cada unidad:")
print("0: quieto, 1: ↑, 2: ↓, 3: ←, 4: →, 5-8: atacar ↑↓←→")

while not done:
    turn += 1
    print(f"\n🔄 Turno {turn} - Equipo {'Jugador (Azul)' if env.current_player == 0 else 'IA (Rojo)'}")

    if env.current_player == 0:
        # Juega el humano (azul)
        action_mask = obs["action_mask"]
        acciones = []
        for i in range(2):
            print(f"Unidad {i} opciones válidas:", np.nonzero(action_mask[i])[0])
            while True:
                try:
                    a = int(input(f"👉 Acción para unidad {i}: "))
                    if a in np.nonzero(action_mask[i])[0]:
                        acciones.append(a)
                        break
                    else:
                        print("❌ Acción inválida según la máscara. Intenta otra.")
                except ValueError:
                    print("❌ Introduce un número entero del 0 al 8.")
    else:
        # Juega la IA (rojo)
        acciones, _ = model.predict(obs, deterministic=True)
        print(f"🤖 IA elige: {acciones}")

    obs, reward, done, _, info = env.step(acciones)

    # Mostrar tablero simplificado
    board_repr = np.full((5, 5), ".")
    for i, unit in enumerate(env.units):
        if unit.is_alive():
            symbol = f"A{i}" if unit.team == 0 else f"R{i-2}"
            x, y = unit.position
            board_repr[x, y] = symbol

    cx, cy = env.capture_point
    if board_repr[cx, cy] == ".":
        board_repr[cx, cy] = "C"

    print("\n".join(" ".join(row) for row in board_repr))
    print(f"Recompensa: {reward:.2f}")

print("\n✅ Partida terminada")
if "episode" in info:
    print(f"Resultado: recompensa {info['episode']['r']:.2f}, turnos: {info['episode']['l']}")
