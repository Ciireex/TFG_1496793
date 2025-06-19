import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Añadir la ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_Knight
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === Wrapper: el rojo está congelado y el azul entrena ===
class FrozenRedKnightEnv(StrategyEnv_Knight):
    def __init__(self, model_red):
        super().__init__()
        self.model_red = model_red

    def step(self, action):
        if self.current_player == 0:
            obs, reward, terminated, truncated, info = super().step(action)
        else:
            obs = self._get_obs()
            red_action, _ = self.model_red.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = super().step(red_action)
            reward = 0  # No recompensar al rojo congelado
        return obs, reward, terminated, truncated, info

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
PREV_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_2soldiers1archer/ppo_2soldiers1archer_final.zip"))
RED_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_red_vs_frozenblue_2s1a/ppo_red_vs_frozenblue_final.zip"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_knight_transfer"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../logs/ppo_knight_transfer"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Cargar modelo rojo congelado ===
print("🔴 Cargando modelo ROJO congelado...")
model_red = PPO.load(RED_MODEL_PATH)
print("✅ Modelo rojo cargado.")

# === Crear entorno vectorizado con wrapper congelado ===
def make_env():
    return FrozenRedKnightEnv(model_red=model_red)

env = DummyVecEnv([make_env for _ in range(4)])

# === Arquitectura del extractor de características ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === Cargar modelo azul desde modelo previo (transfer learning) ===
print("🔵 Cargando modelo AZUL desde checkpoint...")
model = PPO.load(PREV_MODEL_PATH, env=env, device="auto", custom_objects={
    "policy_kwargs": policy_kwargs
})
model.set_env(env)
print("✅ Modelo azul cargado y conectado al nuevo entorno.")

# === Callbacks ===
callbacks = [
    EvalCallback(
        make_env(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_knight_transfer"
    )
]

# === Entrenamiento ===
print("🚀 Iniciando entrenamiento transfer contra ROJO congelado...")
model.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_knight_final.zip"))
print("✅ Entrenamiento finalizado y modelo guardado como ppo_knight_final.zip")
