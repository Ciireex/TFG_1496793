import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase3_Obstaculos
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
MODEL_NAME = "ppo_blue_def_f2"
PRETRAINED_MODEL = "ppo_blue_def_f1"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/" + MODEL_NAME))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === ENTORNO PARA ENTRENAMIENTO ===
env = make_vec_env(Env_Fase3_Obstaculos, n_envs=N_ENVS, seed=456)

# === CARGA MODELO PREENTRENADO Y TRANSFER ===
print("🔄 Cargando modelo preentrenado:", PRETRAINED_MODEL)
model = PPO.load(os.path.join(MODEL_DIR, PRETRAINED_MODEL), env=env, device="auto")

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        Env_Fase3_Obstaculos(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME
    )
]

# === ENTRENAMIENTO ===
print(f"🚀 Continuando entrenamiento PPO azul en Fase 3 ({MODEL_NAME})...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"✅ Modelo PPO azul guardado como {MODEL_NAME}.zip")
