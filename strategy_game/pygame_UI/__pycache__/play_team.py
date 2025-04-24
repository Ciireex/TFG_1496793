import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvTeam import StrategyEnvTeam
from gym_strategy.core.Renderer import Renderer

# Cargar modelo entrenado
model = PPO.load("ppo_strategy_env_team")

# Inicializar entorno
env = StrategyEnvTeam()
obs, _ = env.reset()
renderer = Renderer(width=600, height=600, board_size=(5, 5))

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]
terminated = False

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    move, atk = action

    team = env.turn
    unit = [u for u in env.units if u.team == team and u.is_alive()][env.unit_index % 3]

    print(f"Equipo {team} - Unidad en {unit.position} → Mueve: {DIRECTIONS[move]}, Ataca: {DIRECTIONS[atk]}")

    obs, reward, terminated, truncated, info = env.step(action)

    renderer.draw_board(env.units, blocked_positions=env.blocked_positions)
    time.sleep(0.5)

print("Recompensa final:", reward)
