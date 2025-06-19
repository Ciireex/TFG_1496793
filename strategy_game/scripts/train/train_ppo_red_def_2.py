import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase3_Obstaculos
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
MODEL_NAME = "ppo_red_def_f2"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/" + MODEL_NAME))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blue_def_f2")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === WRAPPER: Azul congelado, rojo aprende ===
class FrozenBlueWrapper(Env_Fase3_Obstaculos):
    def __init__(self):
        super().__init__()
        self.model_blue = PPO.load(BLUE_MODEL_PATH, device="auto")

    def step(self, action):
        if self.current_player == 0:
            obs = self._get_obs()
            action, _ = self.model_blue.predict(obs, deterministic=True)
        return super().step(action)

# === ENTORNO DE ENTRENAMIENTO ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=789)

# === ENTRENAMIENTO PPO ROJO ===
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
        name_prefix=MODEL_NAME
    )
]

# === ENTRENAMIENTO ===
print(f"🚀 Entrenando modelo PPO rojo en Fase 3 ({MODEL_NAME}) vs azul congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"✅ Modelo PPO rojo guardado como {MODEL_NAME}.zip")
