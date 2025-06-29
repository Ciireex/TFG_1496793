import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/ppo/red_f7_v4"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
RED_PREV_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_red_f6_v2.zip")
BLUE_FROZEN_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_f7_v4.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CONFIG POLÍTICA PPO ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === FUNCIONES: Transferir solo capas FC ===
def load_fc_layers_only(model: PPO, path: str):
    data, params, _ = load_from_zip_file(path)
    model_state = model.policy.state_dict()
    filtered_state = {
        k: v for k, v in params.items()
        if k in model_state and model_state[k].shape == v.shape and not k.startswith("features_extractor")
    }
    model.policy.load_state_dict(filtered_state, strict=False)
    print("✅ Pesos transferidos (solo capas pi y vf, sin extractor CNN).")

# === RECONSTRUIR MODELO AZUL CONGELADO (ADAPTADO AL MAPA) ===
print("🧊 Reconstruyendo modelo azul congelado (v4) para Fase 7 Terreno...")
dummy_env = Env_Fase7_Terreno()

frozen_blue = PPO(
    policy="CnnPolicy",
    env=dummy_env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    device="auto"
)
load_fc_layers_only(frozen_blue, BLUE_FROZEN_MODEL_PATH)

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

# === ENTORNO ENVUELTO PARA ENTRENAMIENTO ===
def make_env():
    base_env = Env_Fase7_Terreno()
    return BlueFrozenPolicyWrapper(base_env, blue_model=frozen_blue)

env = make_vec_env(make_env, n_envs=N_ENVS)

# === CREAR MODELO PPO ROJO ===
print("🧠 Creando modelo PPO rojo (F7 v4) con arquitectura nueva...")
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
    seed=45,
    device="auto"
)

# === TRANSFER LEARNING ROJO: Solo capas pi/vf
load_fc_layers_only(model, RED_PREV_MODEL_PATH)

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
        name_prefix="ppo_red_f7_v4"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO rojo en Fase 7 Terreno vs azul congelado F7 v4...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "ppo_red_f7_v4"))
print("✅ Modelo PPO rojo F7 v4 guardado como ppo_red_f7_v4.zip")
