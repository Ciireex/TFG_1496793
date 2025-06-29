import os
import sys
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/maskableppo/blue_f7_v3"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_blue_f6_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER: Red congelado usando acción válida aleatoria ===
class FrozenRedWrapper(Env_Fase7_Terreno):
    def __init__(self):
        super().__init__()

    def step(self, action):
        if self.current_player == 0:  # Entrena azul
            obs, reward, terminated, truncated, info = super().step(action)
        else:  # Red aleatorio válido
            mask = self.get_action_mask()
            action_red = np.flatnonzero(mask)[0] if mask.any() else 0
            obs, reward, terminated, truncated, info = super().step(action_red)
        return obs, reward, terminated, truncated, info

# === MÁSCARA DE ACCIONES ===
def mask_fn(env):
    return env.get_action_mask()

def make_env():
    return ActionMasker(FrozenRedWrapper(), mask_fn)

env = make_vec_env(make_env, n_envs=N_ENVS, seed=42)

# === POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

# === CREACIÓN DEL MODELO (TRANSFER DESDE F6) ===
print("🧠 Creando modelo azul F7_v3 con transferencia desde blue F6...")
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

print("🔁 Cargando pesos del modelo azul F6...")
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
        name_prefix="maskableppo_blue_f7_v3"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando blue F7_v3 contra red congelado en entorno con terreno especial...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "maskableppo_blue_f7_v3"))
print("✅ Modelo azul guardado como maskableppo_blue_f7_v3.zip")
