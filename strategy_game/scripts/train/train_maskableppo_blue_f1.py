import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4

# === IMPORTA TU EXTRACTOR PERSONALIZADO ===
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/maskableppo/blue_f1_v1"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === MÁSCARA DE ACCIÓN ===
def mask_fn(env):
    return env.get_action_mask()

# === ENVOLTORIO ===
def make_maskable_env():
    base_env = Env_Fase1_Soldiers4x4()
    return ActionMasker(base_env, mask_fn)

env = make_vec_env(make_maskable_env, n_envs=N_ENVS, seed=42)

# === PARÁMETROS DEL MODELO ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

model = MaskablePPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    ent_coef=0.01,
    n_steps=2048,
    batch_size=256,
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
    device="auto",
    seed=42
)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        make_maskable_env(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="maskableppo_blue_f1_v1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando MaskablePPO azul Fase 1 con extractor CNN personalizado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO ===
model.save(os.path.join(MODEL_DIR, "maskableppo_blue_f1_v1"))
print("✅ Modelo guardado como maskableppo_blue_f1_v1.zip")
