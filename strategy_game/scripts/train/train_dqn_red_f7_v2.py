import os
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 1
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/dqn/red_f7_v3"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_blue_f7_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === WRAPPER PARA AZUL CONGELADO (DQN) ===
class FrozenBlueWrapper(Wrapper):
    def __init__(self, base_env, blue_model):
        super().__init__(base_env)
        self.blue_model = blue_model

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        while self.env.current_player == 0 and not terminated and not truncated:
            action_blue, _ = self.blue_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action_blue)

        return obs, reward, terminated, truncated, info

# === CARGA DEL MODELO AZUL CONGELADO ===
print("Cargando modelo DQN azul congelado...")
frozen_blue_model = DQN.load(BLUE_MODEL_PATH)

# === CNN PERSONALIZADA PARA DQN ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[256, 128]
)

# === ENTORNO CON WRAPPER (AZUL CONGELADO) ===
env = FrozenBlueWrapper(
    base_env=Env_Fase7_Terreno(),
    blue_model=frozen_blue_model
)

# === CREACIÓN DEL MODELO DQN ROJO ===
print("Creando modelo DQN rojo F7 v3 (vs azul congelado)...")
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="auto"
)

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        eval_env=FrozenBlueWrapper(
            base_env=Env_Fase7_Terreno(),
            blue_model=frozen_blue_model
        ),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15_000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_red_f7_v3"
    )
]

# === ENTRENAMIENTO ===
print("Entrenando modelo DQN rojo en Fase 7 v3 contra DQN azul congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "dqn_red_f7_v3"))
print("Modelo DQN rojo F7 v3 guardado como dqn_red_f7_v3.zip")
