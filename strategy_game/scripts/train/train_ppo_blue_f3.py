import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2Soldiers1Archer_6x4_Obs

# === Configuración general ===
CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_2soldiers1archer"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../logs/ppo_2soldiers1archer"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Parámetros de red CNN y política ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === Entorno vectorizado ===
env = make_vec_env(StrategyEnv_2Soldiers1Archer_6x4_Obs, n_envs=4, seed=42)

# === Crear modelo PPO ===
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

# === Callbacks ===
callbacks = [
    EvalCallback(
        StrategyEnv_2Soldiers1Archer_6x4_Obs(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_2soldiers1archer"
    )
]

# === Entrenar ===
model.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_2soldiers1archer_final.zip"))
print("✅ Modelo guardado como ppo_2soldiers1archer_final.zip")
