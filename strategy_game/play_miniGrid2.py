import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvMiniGrid2 import StrategyEnv1vsDummyRandomSpawn
from gym_strategy.core.Renderer import Renderer

# Cargar modelo entrenado
model = PPO.load("ppo_spawn_v1")  # cambia si usas otro nombre

# Crear entorno
env = StrategyEnv1vsDummyRandomSpawn()
obs, _ = env.reset()

# Inicializar renderizador
renderer = Renderer(width=600, height=600, board_size=(5, 5))
DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]

terminated = False
while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    move, atk = action
    me = env.units[env.turn]
    print(f"IA en {me.position} → Mueve: {DIRECTIONS[move]}, Ataca: {DIRECTIONS[atk]}")

    obs, reward, terminated, truncated, info = env.step(action)
    renderer.draw_board(env.units, blocked_positions=env.blocked_positions)
    time.sleep(0.5)

print("🏁 Recompensa final:", reward)
