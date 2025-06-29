import os
import sys
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_MapaGrande
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 1
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/dqn/blue_f7_v2"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
RED_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_red_f7_v3.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === DEFINICIÓN DEL WRAPPER PARA PPO CONGELADO ===
class FrozenRedWrapper(Wrapper):
    def __init__(self, base_env, red_model, use_mask=False):
        super().__init__(base_env)
        self.red_model = red_model
        self.use_mask = use_mask

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        while self.env.current_player == 1 and not terminated and not truncated:
            mask = info.get("action_mask") if self.use_mask else None
            if self.use_mask:
                red_action, _ = self.red_model.predict(obs, deterministic=True, action_masks=mask)
            else:
                red_action, _ = self.red_model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = self.env.step(red_action)

        return obs, reward, terminated, truncated, info

# === CARGA DEL MODELO PPO ROJO CONGELADO ===
print("📦 Cargando modelo PPO rojo congelado...")
frozen_red_model = PPO.load(RED_MODEL_PATH)

# === POLICY KWARGS PARA CNN PERSONALIZADA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[256, 128]
)

# === ENTORNO PRINCIPAL CON PPO ROJO CONGELADO ===
env = FrozenRedWrapper(
    base_env=Env_Fase7_MapaGrande(),
    red_model=frozen_red_model,
    use_mask=False  # Cambia a True si el modelo rojo es MaskablePPO
)

# === CREACIÓN DEL MODELO DQN AZUL ===
print("🧠 Creando modelo DQN azul Fase 7 v2 desde cero...")
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
        eval_env=FrozenRedWrapper(
            base_env=Env_Fase7_MapaGrande(),
            red_model=frozen_red_model,
            use_mask=False
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
        name_prefix="dqn_blue_f7_v2"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo DQN azul en Fase 7 v2 contra PPO rojo congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "dqn_blue_f7_v2"))
print("✅ Modelo DQN azul F7 v2 guardado como dqn_blue_f7_v2.zip")
