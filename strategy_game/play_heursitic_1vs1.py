import time
import numpy as np
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv1vs1 import StrategyEnv1vs1
from gym_strategy.core.Renderer import Renderer

# Cargar modelo entrenado
model = PPO.load("ppo_strategy_env1vs1")

# Inicializar entorno y renderer
env = StrategyEnv1vs1()
obs, _ = env.reset()
renderer = Renderer(width=600, height=600, board_size=(5, 5))

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]

# Heurística simple: ir hacia el enemigo y atacarlo si está adyacente
def get_heuristic_action(me, enemy):
    dx = enemy.position[0] - me.position[0]
    dy = enemy.position[1] - me.position[1]

    if abs(dx) + abs(dy) == 1:
        if dx == -1: return 0, 3  # atacar izquierda
        if dx == 1: return 0, 4   # atacar derecha
        if dy == -1: return 0, 1  # atacar arriba
        if dy == 1: return 0, 2   # atacar abajo

    if abs(dx) > abs(dy):
        return (3, 0) if dx < 0 else (4, 0)  # moverse en X
    elif dy != 0:
        return (1, 0) if dy < 0 else (2, 0)  # moverse en Y
    return 0, 0  # quieto

turn = env.turn
terminated = False

while not terminated:
    me = env.units[turn]
    enemy = env.units[1 - turn]

    if turn == 0:
        # Recalcular la observación desde la perspectiva de la IA
        obs_model = env._get_obs()[:31]  # Recorta si tu modelo fue entrenado con shape=(31,)
        action, _ = model.predict(obs_model, deterministic=True)
        print(f"IA (Jugador 0 - Azul) → Movimiento: {DIRECTIONS[action[0]]}, Ataque: {DIRECTIONS[action[1]]}")
    else:
        action = get_heuristic_action(me, enemy)
        print(f"Heurística (Jugador 1 - Rojo) → Movimiento: {DIRECTIONS[action[0]]}, Ataque: {DIRECTIONS[action[1]]}")

    obs, reward, terminated, truncated, info = env.step(action)
    turn = 1 - turn

    renderer.draw_board(env.units, blocked_positions=env.blocked_positions)
    time.sleep(0.5)

print("Recompensa final:", reward)
