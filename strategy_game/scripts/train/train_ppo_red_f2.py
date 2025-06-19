import os
import sys
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_3v3Soldiers6x4_Obs
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = "../logs/ppo_red_vs_frozenblue_vf2/"
MODEL_DIR = "../models/ppo_red_vs_frozenblue/"
PRETRAINED_PATH = os.path.join(MODEL_DIR, "ppo_red_vf1.zip")
BLUE_MODEL_PATH = "../models/ppo_blue_vs_heuristicred/ppo_blue_vf2.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === WRAPPER CON AZUL FIJO ===
class FrozenBlueWrapper(StrategyEnv_3v3Soldiers6x4_Obs):
    def __init__(self):
        super().__init__()
        self.model_blue = PPO.load(BLUE_MODEL_PATH)

    def step(self, action):
        if self.current_player == 0:
            obs = self._get_obs()
            action, _ = self.model_blue.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)

# === ENV ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=888)

# === TRANSFERENCIA DE PESOS ===
print("🧠 Cargando modelo anterior ppo_red_vf1...")
model_old = PPO.load(PRETRAINED_PATH)

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
    seed=888,
    device="auto"
)

model.set_parameters(model_old.get_parameters())
print("✅ Pesos cargados desde ppo_red_vf1.zip")

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
        name_prefix="ppo_red_vf2"
    )
]

# === ENTRENAMIENTO ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_red_vf2"))
print("✅ Modelo rojo entrenado guardado como ppo_red_vf2.zip")
