import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase5_Knight
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN GENERAL ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/ppo/red_f5"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
RED_PREV_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_red_f4_v1.zip")
BLUE_FROZEN_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f5_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER: BlueFrozenPolicyWrapper ===
class BlueFrozenPolicyWrapper(Wrapper):
    def __init__(self, env, blue_model):
        super().__init__(env)
        self.blue_model = blue_model

    def step(self, action):
        if self.env.current_player == 1:
            obs, reward, done, trunc, info = self.env.step(action)
        else:
            action_blue, _ = self.blue_model.predict(self.env._get_obs(), deterministic=True)
            obs, reward, done, trunc, info = self.env.step(action_blue)
        return obs, reward, done, trunc, info

# === CARGAR MODELO AZUL CONGELADO ===
print("🧊 Cargando modelo azul congelado F5...")
frozen_blue = PPO.load(BLUE_FROZEN_MODEL_PATH, device="auto")

# === CREAR ENTORNO ENVUELTO PARA ENTRENAMIENTO ===
def make_env():
    base_env = Env_Fase5_Knight()
    return BlueFrozenPolicyWrapper(base_env, blue_model=frozen_blue)

env = make_vec_env(make_env, n_envs=N_ENVS)

# === CONFIGURACIÓN DE LA POLÍTICA PPO ROJO ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

print("🧠 Creando modelo PPO rojo (F5) con pesos de F4...")
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
    seed=43,
    device="auto"
)

model.set_parameters(RED_PREV_MODEL_PATH, exact_match=False)

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
        name_prefix="ppo_red_f5_v1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO rojo en Fase 5 vs azul congelado F5...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDAR MODELO ENTRENADO ===
model.save(os.path.join(MODEL_DIR, "ppo_red_f5_v1"))
print("✅ Modelo PPO rojo F5 guardado como ppo_red_f5_v1.zip")
