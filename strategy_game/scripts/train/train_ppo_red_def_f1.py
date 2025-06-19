import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase2_Soldiers6x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/ppo_red_def_f1"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_def_f1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGAR MODELO AZUL ===
frozen_blue = PPO.load(BLUE_MODEL_PATH)

# === WRAPPER: Azul congelado, Rojo aprende ===
class FrozenBlueWrapper(Env_Fase2_Soldiers6x4):
    def step(self, action):
        if self.current_player == 0:  # Azul (congelado)
            obs = self._get_obs()
            act, _ = frozen_blue.predict(obs, deterministic=True)
            return super().step(act)
        else:  # Rojo (modelo en entrenamiento)
            return super().step(action)

# === ENTORNO ENVOLTORIO ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=789)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === ENTRENAMIENTO PPO ===
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
    seed=789,
    device="auto"
)

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
        name_prefix="ppo_red_def_f1"
    )
]

# === EJECUTAR ENTRENAMIENTO ===
print("🚩 Entrenando modelo PPO rojo en Fase 2 contra azul congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_red_def_f1"))
print("✅ Modelo PPO rojo guardado como ppo_red_def_f1.zip")
