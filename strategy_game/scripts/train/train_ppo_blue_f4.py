import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase4_Archer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN GENERAL ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/ppo/blue_f4"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
BLUE_PREV_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f3_v3.zip")
RED_FROZEN_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_red_f3_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER: RedFrozenPolicyWrapper ===
class RedFrozenPolicyWrapper(Wrapper):
    def __init__(self, env, red_model):
        super().__init__(env)
        self.red_model = red_model

    def step(self, action):
        if self.env.current_player == 0:
            obs, reward, done, trunc, info = self.env.step(action)
        else:
            action_red, _ = self.red_model.predict(self.env._get_obs(), deterministic=True)
            obs, reward, done, trunc, info = self.env.step(action_red)
        return obs, reward, done, trunc, info

# === CARGAR MODELO ROJO CONGELADO ===
print("🧊 Cargando modelo rojo congelado F3...")
frozen_red = PPO.load(RED_FROZEN_MODEL_PATH, device="auto")

# === CREAR ENTORNO ENVUELTO PARA ENTRENAMIENTO ===
def make_env():
    base_env = Env_Fase4_Archer()
    return RedFrozenPolicyWrapper(base_env, red_model=frozen_red)

env = make_vec_env(make_env, n_envs=N_ENVS)

# === CONFIGURACIÓN DE LA POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === CREAR Y CARGAR MODELO PPO AZUL ===
print("🧠 Creando modelo PPO azul (F4) con pesos de F3...")
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

model.set_parameters(BLUE_PREV_MODEL_PATH, exact_match=False)

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
        name_prefix="ppo_blue_f4_v1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO azul en Fase 4 vs rojo congelado F3...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDAR MODELO ENTRENADO ===
model.save(os.path.join(MODEL_DIR, "ppo_blue_f4_v1"))
print("✅ Modelo PPO azul F4 guardado como ppo_blue_f4_v1.zip")
