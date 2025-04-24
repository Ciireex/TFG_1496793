import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv1vsDummy import StrategyEnv1vsDummy
from gym_strategy.core.Renderer import Renderer

# Cargar el modelo entrenado
model = PPO.load("ppo_strategy_env1vsdummy_multiinput")

# Crear entorno y renderer
env = StrategyEnv1vsDummy()
obs, _ = env.reset()
renderer = Renderer(width=600, height=600, board_size=(5, 5))

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]
terminated = False

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    move, atk = action

    me = env.units[env.turn]
    print(f"IA (Jugador {me.team}) en {me.position} → Mueve: {DIRECTIONS[move]}, Ataca: {DIRECTIONS[atk]}")

    obs, reward, terminated, truncated, info = env.step(action)

    renderer.draw_board(env.units, blocked_positions=env.blocked_positions)
    time.sleep(0.5)

print("Recompensa final:", reward)
