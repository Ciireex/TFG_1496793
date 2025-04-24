# play_combat.py

import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_strategy.envs.StrategyEnvCombat import StrategyEnvCombat

# 1) Creamos un DummyVecEnv con una única copia real del entorno
def make_env():
    return StrategyEnvCombat()

venv = DummyVecEnv([make_env])

# 2) Cargamos los parámetros de normalización
venv = VecNormalize.load("models/ppo_combat_5_vecnormalize.pkl", venv)
venv.training    = False  # no seguir actualizando medias/varianzas
venv.norm_reward = False  # no escalar recompensas en evaluación

# 3) Cargamos el modelo, pasándole el vecenv normalizado
model = PPO.load("models/ppo_combat_5.zip", env=venv)

# 4) Hacemos el reset del vecenv (Gymnasium devuelve obs, info)
obs = venv.reset()
done = False

# Mapas para imprimir acciones
MOVE = ["Sin mover", "Mover"]
ACT  = ["Atacar",    "Pasar"]

while not done:
    # 5) model.predict nos devuelve action con forma (1, 4)
    action_array, _ = model.predict(obs, deterministic=True)
    mv, dx_i, dy_i, act_type = action_array[0]

    # 6) Reconstruimos el desplazamiento real
    dx = int(dx_i) - StrategyEnvCombat.MAX_D
    dy = int(dy_i) - StrategyEnvCombat.MAX_D

    # 7) Para imprimir, sacamos la unidad activa y su destino
    env0 = venv.envs[0]
    unit = env0.turn_units[env0.unit_index]
    dest = (unit.position[0] + dx, unit.position[1] + dy)

    # 8) Mensaje amigable
    print(
        f"Equipo {env0.current_turn} → "
        f"{MOVE[mv]}{' → '+str(dest) if mv else ''}, "
        f"{ACT[act_type]}"
    )

    # 9) Ejecutamos el paso y desempacamos las 5 tuplas de Gymnasium
    obs, reward, terminated, truncated, _ = venv.step([action_array[0]])
    done = bool(terminated or truncated)

    time.sleep(0.4)

print("🏁 Partida finalizada.")
