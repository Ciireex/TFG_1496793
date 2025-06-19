
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === Wrapper: Azul actúa con política entrenada, rojo aprende ===
class FrozenBlueWrapper(StrategyEnv_2v2Soldiers4x4):
    def __init__(self, blue_model_path):
        super().__init__()
        self.blue_model = PPO.load(blue_model_path)
        self.current_action = None

    def step(self, action):
        if self.current_player == 0:
            obs = self._get_obs()
            self.current_action, _ = self.blue_model.predict(obs, deterministic=True)
            return super().step(self.current_action)
        else:
            return super().step(action)

# === Configuración general ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
SEED = 77
LOG_DIR = "./logs/ppo_red_vs_frozenblue/"
MODEL_DIR = "./models/ppo_red_vs_frozenblue/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Política personalizada (CNN mejorada) ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === Crear entornos vectorizados ===
blue_model_path = "./models/ppo_blue_vs_randomred/ppo_blue_f0_v0.zip"

env = make_vec_env(
    lambda: FrozenBlueWrapper(blue_model_path),
    n_envs=N_ENVS,
    seed=SEED
)

# === Callbacks ===
eval_callback = EvalCallback(
    FrozenBlueWrapper(blue_model_path),
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=15000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=MODEL_DIR,
    name_prefix="ppo_red"
)

# === Modelo PPO con CNN táctica ===
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
    seed=SEED,
    device="auto"
)

# === Entrenamiento ===
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# === Guardar modelo final ===
model.save(os.path.join(MODEL_DIR, "ppo_red_f0_v1"))
print("✅ Entrenamiento de PPO rojo contra azul congelado completado.")
