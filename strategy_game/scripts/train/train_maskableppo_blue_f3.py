import os
import sys
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase3_Obstaculos
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/maskableppo/blue_f3"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_blue_f2_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === MÁSCARA DE ACCIONES ===
def mask_fn(env):
    return env.get_action_mask()

def make_env():
    base_env = Env_Fase3_Obstaculos()
    return ActionMasker(base_env, mask_fn)

env = make_vec_env(make_env, n_envs=N_ENVS, seed=42)

# === CONFIGURACIÓN DE LA POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

# === CREACIÓN DEL NUEVO MODELO (estructura nueva para Fase 3) ===
print("🧠 Creando nuevo modelo MaskablePPO azul para Fase 3...")
model = MaskablePPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    seed=42,
    device="auto"
)

# === TRANSFER LEARNING REAL ===
print("🔁 Cargando pesos del modelo azul F2 (transfer learning)...")
model.set_parameters(PREV_MODEL_PATH, exact_match=False)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        make_env(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="maskableppo_blue_f3_v1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando MaskablePPO azul F3 con transferencia desde F2...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO ===
model.save(os.path.join(MODEL_DIR, "maskableppo_blue_f3_v1"))
print("✅ Modelo azul guardado como maskableppo_blue_f3_v1.zip")
