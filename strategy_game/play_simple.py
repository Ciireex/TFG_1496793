import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvSimple import StrategyEnvSimple

env   = StrategyEnvSimple()
model = PPO.load("models/ppo_captura.zip")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    move_flag, dx_i, dy_i, act_type = action
    dx, dy = dx_i - env.MAX_D, dy_i - env.MAX_D          # Δ reales

    # ─── Mostrar intención ────────────────────────────
    print(f"Equipo {env.current_turn} → ", end="")
    if move_flag:
        dest = (env.turn_units[env.unit_index].position[0] + dx,
                env.turn_units[env.unit_index].position[1] + dy)
        print(f"Mover → {dest}, ", end="")
    else:
        print("Sin mover, ", end="")

    print(["Atacar", "Capturar", "Pasar"][act_type])

    # Un paso
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.5)

print("🏁 Partida finalizada.")
