import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase2_Soldiers6x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/ppo/blue_f2"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f1_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === ENTORNO NUEVO 6x4 ===
env = make_vec_env(Env_Fase2_Soldiers6x4, n_envs=N_ENVS, seed=42)

# === CONFIG POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === CREACIÓN DEL NUEVO MODELO (estructura nueva con obs 21x6x4) ===
print("🧠 Creando nuevo modelo PPO azul para entorno 6x4...")
model = PPO(
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
print("🔁 Cargando pesos del modelo PPO azul F1 v3...")
model.set_parameters(PREV_MODEL_PATH, exact_match=False)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        Env_Fase2_Soldiers6x4(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_blue_f2_v3"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO azul en Fase 2 (6x4)...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDAR ===
model.save(os.path.join(MODEL_DIR, "ppo_blue_f2_v3"))
print("✅ Modelo PPO azul F2 guardado como ppo_blue_f2_v3.zip")
