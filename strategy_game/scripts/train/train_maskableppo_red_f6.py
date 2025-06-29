import os
import sys
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase6_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/maskableppo/red_f6"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_red_f5_v1.zip")
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "maskableppo_blue_f6_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === ENV WRAPPER PARA CONGELAR EQUIPO AZUL ===
class FrozenBlueWrapper(Env_Fase6_Terreno):
    def __init__(self, model_path):
        super().__init__()
        self.blue_model = MaskablePPO.load(model_path, device="cpu")  # O usa "auto" si tienes GPU
        self.current_obs = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action):
        if self.current_player == 1:  # Turno del equipo azul (congelado)
            with np.errstate(all='ignore'):
                mask = self.get_action_mask()
                if mask.sum() == 0:
                    action_blue = 0  # Acción por defecto si no hay válidas
                else:
                    action_blue, _ = self.blue_model.predict(self.current_obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = super().step(action_blue)
            self.current_obs = obs
        else:  # Turno del equipo rojo (a entrenar)
            obs, reward, terminated, truncated, info = super().step(action)
            self.current_obs = obs
        return obs, reward, terminated, truncated, info

# === MÁSCARA DE ACCIONES ===
def mask_fn(env):
    return env.get_action_mask()

def make_env():
    env = FrozenBlueWrapper(BLUE_MODEL_PATH)
    return ActionMasker(env, mask_fn)

env = make_vec_env(make_env, n_envs=N_ENVS, seed=42)

# === CONFIGURACIÓN DE LA POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

# === CREACIÓN DEL MODELO ROJO (desde F5) ===
print("🧠 Cargando modelo rojo MaskablePPO desde Fase 5...")
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
        name_prefix="maskableppo_red_f6_v1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Continuando entrenamiento del modelo rojo F6 contra el azul F6 congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO ===
model.save(os.path.join(MODEL_DIR, "maskableppo_red_f6_v1"))
print("✅ Modelo rojo guardado como maskableppo_red_f6_v1.zip")
