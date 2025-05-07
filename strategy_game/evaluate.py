import sys, os
sys.path.append(os.path.abspath("."))

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import A2C

from gym_strategy.envs.Env3v3 import Env3v3
from gym_strategy.core.Unit import Soldier, Archer

# Configuración
MODEL_NAME = "ppo_3v3_soldiers_archers"  # o "a2c_3v3_soldiers_archers"
USE_MASKABLE = MODEL_NAME.startswith("ppo")
NUM_EPISODES = 1000

def mask_fn(env):
    return env._get_action_mask()

# Equipos
blue_team = [Soldier, Soldier, Archer]
red_team = [Archer, Soldier, Soldier]

# Cargar modelo
if USE_MASKABLE:
    model = MaskablePPO.load(MODEL_NAME)
else:
    model = A2C.load(MODEL_NAME)

# Crear entorno
def make_env():
    base_env = Env3v3(blue_team=blue_team, red_team=red_team)
    return ActionMasker(base_env, mask_fn) if USE_MASKABLE else base_env

env = make_env()

# Métricas
wins_blue = 0
wins_red = 0
draws = 0

for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    done = False
    current_player = 0

    while not done:
        current_player = env.env.current_player  # Quién juega ahora
        mask = obs["action_mask"] if USE_MASKABLE else None
        action, _ = model.predict(obs, deterministic=True, action_masks=mask) if USE_MASKABLE else model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

    # Evaluar resultado
    if "episode" in info:
        if info["episode"]["r"] == 1.5:
            if current_player == 0:
                wins_blue += 1
            else:
                wins_red += 1
        else:
            draws += 1

    if episode % 100 == 0:
        print(f"✔️ {episode} partidas simuladas...")

# Resultados
print("\nRESULTADOS FINALES")
print(f"Victorias equipo azul: {wins_blue} / {NUM_EPISODES} ({(wins_blue / NUM_EPISODES) * 100:.2f}%)")
print(f"Victorias equipo rojo: {wins_red} / {NUM_EPISODES} ({(wins_red / NUM_EPISODES) * 100:.2f}%)")
print(f"Empates / sin resolución clara: {draws} ({(draws / NUM_EPISODES) * 100:.2f}%)")
