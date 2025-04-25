import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvCaptureSparse import StrategyEnvCaptureSparse
from gym_strategy.core.Renderer import Renderer

# Cargar modelo entrenado
model = PPO.load("ppo_capture_sparse")

# Crear entorno
env = StrategyEnvCaptureSparse()
obs, _ = env.reset()

# Inicializar render
renderer = Renderer(width=600, height=600, board_size=(5, 5))
DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]

terminated = False
turn = 0

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    move, _ = action

    me = env.units[env.turn]
    print(f"IA en {me.position} → Movimiento: {DIRECTIONS[move]}")

    obs, reward, terminated, truncated, info = env.step(action)

    renderer.draw_board(
        env.units,
        blocked_positions=env.blocked_positions,
        capture_point=env.capture_point
    )

    time.sleep(0.5)

print("🏁 Recompensa final:", reward)
