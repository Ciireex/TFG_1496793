import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Añadir ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/a2c/red_f1_v3"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
FROZEN_BLUE_PATH = os.path.join(MODEL_DIR, "a2c_blue_f1_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === ENVOLTORIO CON POLÍTICA AZUL CONGELADA ===
class FrozenBlueWrapper(Env_Fase1_Soldiers4x4):
    def __init__(self):
        super().__init__()
        self.frozen_blue = A2C.load(FROZEN_BLUE_PATH, custom_objects={
            "features_extractor_class": EnhancedTacticalFeatureExtractor
        })

    def step(self, action):
        if self.current_player == 0:  # azul (congelado)
            obs = self._get_obs()
            action, _ = self.frozen_blue.predict(obs, deterministic=True)
            return super().step(action)
        else:  # rojo (aprende)
            return super().step(action)

# === CREAR ENTORNO VECTORIAL ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=456)

# === ENTRENAMIENTO ===
model = A2C(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    max_grad_norm=0.5,
    seed=456,
    device="auto"
)

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
        name_prefix="a2c_red_f1_v3"
    )
]

# === INICIO ENTRENAMIENTO ===
print("🚀 Entrenando modelo A2C rojo F1 v3 (contra azul congelado)...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "a2c_red_f1_v3"))
print("✅ Modelo A2C rojo guardado como a2c_red_f1_v3.zip")
