import os
import sys
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/ppo/red_f1_v3"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f1_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === CARGA DEL MODELO AZUL CONGELADO ===
blue_model = PPO.load(BLUE_MODEL_PATH, custom_objects={
    "features_extractor_class": EnhancedTacticalFeatureExtractor
})

# === ENVOLTORIO CON PPO AZUL CONGELADO ===
class BlueFrozenWrapper(Env_Fase1_Soldiers4x4):
    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:  # modelo azul congelado
            action, _ = blue_model.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)

# === ENTORNO DE ENTRENAMIENTO ===
env = make_vec_env(lambda: BlueFrozenWrapper(), n_envs=N_ENVS, seed=42)

# === MODELO PPO ROJO ===
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

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        BlueFrozenWrapper(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_red_f1_v3"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO rojo en Fase 1 (4x4) vs azul congelado v3...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "ppo_red_f1_v3"))
print("✅ Modelo PPO rojo guardado como ppo_red_f1_v3.zip")
