import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_device

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase3_Obstaculos
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/a2c/blue_f3_v4"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREVIOUS_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_f2_v1.zip")  # <<--- TRANSFER
RED_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_red_f3_v2.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER: ROJO CON POLÍTICA FIJA ===
class FrozenRedWrapper(Env_Fase3_Obstaculos):
    def __init__(self):
        super().__init__()
        self.red_model = A2C.load(RED_MODEL_PATH, device="auto")

    def step(self, action):
        if self.current_player == 1:
            obs = self._get_obs()
            action, _ = self.red_model.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)

# === ENTORNO ===
env = make_vec_env(lambda: FrozenRedWrapper(), n_envs=4, seed=42)

# === POLÍTICA PERSONALIZADA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === MODELO A2C Y TRANSFER DESDE F2 ===
model = A2C(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,         # más bajo para fine-tuning estable
    n_steps=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,
    max_grad_norm=0.5,
    seed=42,
    device="auto"
)

# === CARGAR PESOS DE F2 ===
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
        name_prefix="a2c_blue_f3_v4"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Fine-tuning A2C azul F3 v4 desde F2 (con entorno aleatorio y mejoras)...")
model.learn(total_timesteps=1_000_000, callback=callbacks, progress_bar=True)

# === GUARDAR ===
model.save(os.path.join(MODEL_DIR, "a2c_blue_f3_v4"))
print("✅ Modelo F3 v4 azul guardado.")
