import os
import sys
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_device

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/a2c/blue_f7_v2"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREVIOUS_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_f6_v1.zip")
RED_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_red_f6_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER: Rojo congelado con acción válida aleatoria
class FrozenRedWrapper(Env_Fase7_Terreno):
    def __init__(self):
        super().__init__()

    def step(self, action):
        if self.current_player == 0:  # Azul entrena
            return super().step(action)
        else:
            mask = self.get_action_mask()
            action_red = np.flatnonzero(mask)[0] if mask.any() else 0
            return super().step(action_red)

# === ENV VECTORIAL ===
env = make_vec_env(lambda: FrozenRedWrapper(), n_envs=4, seed=42)

# === POLÍTICA PERSONALIZADA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === CREACIÓN DEL MODELO AZUL F7 V2 CON TRANSFER ===
model = A2C(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,
    max_grad_norm=0.5,
    seed=42,
    device="auto"
)

print("🔁 Transferencia desde F6 (solo pesos de la política)...")
model_old = A2C.load(PREVIOUS_MODEL_PATH, device=get_device("auto"))
model.policy.load_state_dict(model_old.policy.state_dict(), strict=False)

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
        name_prefix="a2c_blue_f7_v2"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando A2C azul F7 v2 (contra rojo aleatorio congelado)...")
model.learn(total_timesteps=1_000_000, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "a2c_blue_f7_v2"))
print("✅ Modelo azul F7 v2 guardado.")
