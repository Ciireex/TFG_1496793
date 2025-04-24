import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnv1vs1 import StrategyEnv1vs1
from gym_strategy.core.Renderer import Renderer

# Cargar modelo entrenado
model = PPO.load("ppo_strategy_env1vs1_v3")

# Inicializar entorno
env = StrategyEnv1vs1()
obs, _ = env.reset()
renderer = Renderer(width=600, height=600, board_size=(5, 5))

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]
terminated = False
turn = env.turn

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    move, atk = action

    # Info
    me = env.units[turn]
    print(f"Jugador {turn} (pos: {me.position}) → Mueve: {DIRECTIONS[move]}, Ataca: {DIRECTIONS[atk]}")

    # Aplicar acción
    obs, reward, terminated, truncated, info = env.step(action)
    turn = 1 - turn

    # Dibujar
    renderer.draw_board(env.units, blocked_positions=env.blocked_positions)
    time.sleep(0.5)

print("Recompensa final:", reward)
