import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase2_Soldiers6x4_Obst
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/a2c_blue_def_f1_retrain2"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
RED_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_red_def_f1_retrain.zip")
PREV_BLUE_PATH = os.path.join(MODEL_DIR, "a2c_blue_def_f1_retrain.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGA DEL MODELO ROJO FIJO ===
red_model = A2C.load(RED_MODEL_PATH)

# === WRAPPER: ENTORNO CON ROJO CONGELADO ===
class FrozenRedWrapper(Env_Fase2_Soldiers6x4_Obst):
    def step(self, action):
        if self.current_player == 1:  # rojo actúa con política fija
            obs = self._get_obs()
            action, _ = red_model.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)

# === ENV ENTRENAMIENTO AZUL ===
env = make_vec_env(lambda: FrozenRedWrapper(), n_envs=N_ENVS, seed=999)

# === CARGA DEL MODELO AZUL ANTERIOR ===
model = A2C.load(PREV_BLUE_PATH, env=env, device="auto")
model.tensorboard_log = LOG_DIR
model.set_parameters(PREV_BLUE_PATH, exact_match=True)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        FrozenRedWrapper(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="a2c_blue_def_f1_retrain2"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando A2C azul (retrain2) contra rojo congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "a2c_blue_def_f1_retrain2"))
print("✅ Modelo A2C azul guardado como a2c_blue_def_f1_retrain2.zip")
