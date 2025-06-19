import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_3v3Soldiers6x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = "../logs/ppo_red_vs_frozenblue_vf1/"
MODEL_DIR = "../models/ppo_red_vs_frozenblue/"
BLUE_MODEL_PATH = "../models/ppo_blue_vs_heuristicred/ppo_blue_vf1.zip"
PRETRAINED_PATH = "../models/ppo_red_vs_frozenblue/ppo_red_vf0.zip"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === WRAPPER CON MODELO AZUL CONGELADO ===
class FrozenBlueWrapper(StrategyEnv_3v3Soldiers6x4):
    def __init__(self, blue_model_path):
        super().__init__()
        self.blue_model = PPO.load(blue_model_path)

    def step(self, action):
        if self.current_player == 0:  # azul congelado
            obs = self._get_obs()
            frozen_action, _ = self.blue_model.predict(obs, deterministic=True)
            return super().step(frozen_action)
        else:
            return super().step(action)

env = make_vec_env(lambda: FrozenBlueWrapper(BLUE_MODEL_PATH), n_envs=N_ENVS, seed=456)

# === TRANSFER LEARNING DESDE ROJO V0 ===
print("🧠 Cargando modelo rojo anterior...")
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
    seed=456,
    device="auto"
)

model.set_parameters(model_old.get_parameters())
print("✅ Pesos rojos transferidos desde ppo_red_vf0.zip")

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        FrozenBlueWrapper(BLUE_MODEL_PATH),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_red_vf1"
    )
]

# === ENTRENAMIENTO ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_red_vf1"))
print("✅ Modelo rojo guardado como ppo_red_vf1.zip")
