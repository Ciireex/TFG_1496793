import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
CURRENT_DIR = os.path.dirname(__file__)
MODEL_PATH_BLUE = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_blue_def_f0.zip"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../logs/ppo_red_def_f0"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGA DEL MODELO AZUL ENTRENADO ===
print("🧠 Cargando modelo PPO azul fijo...")
model_blue = PPO.load(MODEL_PATH_BLUE)
print("✅ Modelo azul cargado.")

# === WRAPPER: Azul fijo, rojo aprende ===
class FrozenBlueWrapper(Env_Fase1_Soldiers4x4):
    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:  # azul = fijo
            action, _ = model_blue.predict(obs, deterministic=True)
            return super().step(action)
        else:
            return super().step(action)  # rojo = aprende

# === CREACIÓN DEL ENTORNO VECORIZADO ===
env = make_vec_env(lambda: FrozenBlueWrapper(), n_envs=N_ENVS, seed=456)

# === PARÁMETROS DE LA RED ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === MODELO PPO ROJO ===
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
        name_prefix="ppo_red_def_f0"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO rojo contra azul fijo...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO ===
model.save(os.path.join(MODEL_DIR, "ppo_red_def_f0"))
print("✅ Modelo PPO rojo guardado como ppo_red_def_f0.zip")
