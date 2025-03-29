from gym_strategy.envs.StrategyEnv import StrategyEnv
import pygame
import time

env = StrategyEnv()
obs = env.reset()
done = False

turn_number = 1
turno_equipo = 0

print("\n🏁 Inicio de la partida\n")

while not done:
    print(f"🔢 Turno {turn_number} - Equipo {turno_equipo}")

    # Acción aleatoria
    action = env.action_space.sample()
    action_type, x, y = env.decode_action(action)
    tipo_str = "Mover" if action_type == 0 else "Atacar"

    print(f"🕹️ Acción: {tipo_str} a posición ({x}, {y})")

    # Aplicar acción
    obs, reward, done, _ = env.step(action)
    print(f"🎯 Recompensa: {reward}\n")

    # Actualizar eventos de pygame
    pygame.event.pump()

    # Esperar para ver mejor el movimiento
    time.sleep(0.6)

    # Avanzar turno visual (aunque el entorno lleva internamente el control)
    turn_number += 1
    turno_equipo = 1 - turno_equipo

# Determinar ganador
team_0_alive = any(u.team == 0 for u in env.units)
team_1_alive = any(u.team == 1 for u in env.units)

if team_0_alive and not team_1_alive:
    winner = 0
elif team_1_alive and not team_0_alive:
    winner = 1
else:
    winner = "Nadie (empate raro)"

print(f"\n🎇 ¡Fin de la partida! Ganador: Equipo {winner} 🎉")
