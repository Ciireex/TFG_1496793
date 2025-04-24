import os
import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvFase1 import StrategyEnvFase1

# Configuración
MODEL_PATH_0 = "models/ppo_fase1.zip"
MODEL_PATH_1 = "models/ppo_fase1.zip"
SLEEP_TIME = 0.4

# Diccionarios de ayuda para mostrar direcciones
dir_labels = ["quieto", "↑", "↓", "←", "→"]

def main():
    print("🎮 Cargando modelos...")
    model_0 = PPO.load(MODEL_PATH_0)
    model_1 = PPO.load(MODEL_PATH_1)

    env = StrategyEnvFase1()
    env.use_knights = False
    env.use_archers = False
    env.use_capture = False
    obs, _ = env.reset()

    done = False
    truncated = False

    while not (done or truncated):
        current_team = env.current_turn
        model = model_0 if current_team == 0 else model_1
        action, _ = model.predict(obs, deterministic=True)
        move_dir, act_mode, act_dir = action

        # Unidad activa
        team_units = [u for u in env.units if u.team == current_team]
        if env.unit_index < len(team_units):
            unit = team_units[env.unit_index]
            unit_type = unit.unit_type
            position = unit.position
        else:
            unit_type = "?"
            position = "?"

        action_name = {0: "Pasar", 1: "Atacar", 2: "Capturar"}.get(act_mode, "???")

        print(f"Equipo {current_team} – {unit_type} en {position} → mueve {dir_labels[move_dir]}, acción: {action_name} {dir_labels[act_dir]}")

        obs, reward, done, truncated, info = env.step(action)
        time.sleep(SLEEP_TIME)

    print("🏁 Partida finalizada.")

if __name__ == "__main__":
    main()
