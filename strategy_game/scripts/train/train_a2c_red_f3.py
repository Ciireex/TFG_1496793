import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase2_Soldiers6x4_Obst
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/a2c_red_def_f1_retrain"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_def_f1_retrain.zip")
PREV_RED_PATH = os.path.join(MODEL_DIR, "a2c_red_def_f1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGA DEL MODELO AZUL FIJO ===
blue_model = A2C.load(BLUE_MODEL_PATH)

# === WRAPPER: ENTORNO CON AZUL CONGELADO ===
class FrozenBlueWrapper(Env_Fase2_Soldiers6x4_Obst):
    def step(self, action):
        if self.current_player == 0:  # azul actúa con política fija
            obs = self._get_obs()
            action, _ = blue_model.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)

# === ENV ENTRENAMIENTO ROJO ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=789)

# === TRANSFER LEARNING DESDE MODELO ROJO PREVIO ===
model = A2C.load(PREV_RED_PATH, env=env, device="auto")
model.tensorboard_log = LOG_DIR
model.set_parameters(PREV_RED_PATH, exact_match=True)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        FrozenBlueWrapper(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="a2c_red_def_f1_retrain"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando A2C rojo contra azul congelado en Fase 2...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "a2c_red_def_f1_retrain"))
print("✅ Modelo A2C rojo mejorado guardado como a2c_red_def_f1_retrain.zip")
