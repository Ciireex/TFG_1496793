# play_combat.py
import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvCombat import StrategyEnvCombat   # ← entorno de combate puro

# 1)  Crea el entorno “fase 1”
env = StrategyEnvCombat()

# 2)  Carga el modelo entrenado sólo para combate
model = PPO.load("models/ppo_combat.zip")

# 3)  Inicia partida
obs, _ = env.reset()
done = False

MOVE_NAMES = ["Sin mover", "Mover"]
ACT_NAMES  = ["Atacar", "Pasar"]          # (no hay Capturar en esta fase)

while not done:
    # 4)  Predicción determinista
    action, _ = model.predict(obs, deterministic=True)
    move_flag, dx_i, dy_i, act_type = action
    dx, dy = dx_i - env.MAX_D, dy_i - env.MAX_D        # Δ reales −3…+3

    # 5)  Mensaje amigable
    unit = env.turn_units[env.unit_index]
    dest = (unit.position[0] + dx, unit.position[1] + dy)
    print(f"Equipo {env.current_turn}  →  "
          f"{MOVE_NAMES[move_flag]} {'' if not move_flag else f'→ {dest} , '} "
          f"{ACT_NAMES[act_type]}")

    # 6)  Avanza el entorno
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.4)

print("🏁  Partida finalizada.")
