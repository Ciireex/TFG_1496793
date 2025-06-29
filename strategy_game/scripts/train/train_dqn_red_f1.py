import os
import sys
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/dqn/red_f1"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
FROZEN_BLUE_PATH = os.path.join(MODEL_DIR, "dqn_blue_f1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[256, 128]  # DQN solo usa una lista
)

# === ENVOLTORIO CON POLÍTICA AZUL CONGELADA (DQN ROJO vs AZUL congelado) ===
class FrozenBlueWrapper(Env_Fase1_Soldiers4x4):
    def __init__(self):
        super().__init__()
        self.frozen_blue = DQN.load(FROZEN_BLUE_PATH, custom_objects={
            "features_extractor_class": EnhancedTacticalFeatureExtractor
        })

    def step(self, action):
        if self.current_player == 0:  # azul (congelado)
            obs = self._get_obs()
            action, _ = self.frozen_blue.predict(obs, deterministic=True)
            return super().step(action)
        else:  # rojo (aprende)
            return super().step(action)

# === CREAR ENTORNO NO VECTORIAL ===
env = FrozenBlueWrapper()

# === MODELO DQN (ROJO) ===
model = DQN(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    gamma=0.99,
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
        name_prefix="dqn_red_f1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo DQN rojo F1 (contra azul congelado)...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "dqn_red_f1"))
print("✅ Modelo DQN rojo guardado como dqn_red_f1.zip")
