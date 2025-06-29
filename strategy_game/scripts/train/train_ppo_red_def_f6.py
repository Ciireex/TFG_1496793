import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Añadir ruta del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase6_MapaGrande
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
MODEL_NAME = "ppo_red_def_f6"
PRETRAINED_RED = "ppo_red_def_f4"
BLUE_MODEL = "ppo_blue_def_f6"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/" + MODEL_NAME))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Wrapper personalizado con azul congelado ===
class BlueFrozenWrapper_Fase6(Env_Fase6_MapaGrande):
    def __init__(self, blue_model):
        super().__init__()
        self.blue_model = blue_model

    def step(self, action):
        if self.current_player == 1:
            # Rojo entrena
            obs, reward, terminated, truncated, info = super().step(action)
        else:
            # Azul congelado (ya entrenado en mapa grande)
            obs = self._get_obs()
            action, _ = self.blue_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

# === Cargar modelo azul ya entrenado en mapa grande ===
print("🔵 Cargando modelo PPO azul:", BLUE_MODEL)
model_blue = PPO.load(os.path.join(MODEL_DIR, BLUE_MODEL), device="auto")

# === Crear entorno Dummy con wrapper ===
def make_env():
    return BlueFrozenWrapper_Fase6(model_blue)

env = DummyVecEnv([make_env])

# === Transfer learning desde rojo fase 4 ===
print("🔴 Aplicando transfer learning desde:", PRETRAINED_RED)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

model_red = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
    device="auto"
)

# Cargar pesos del modelo anterior (Fase 4)
pretrained_model = PPO.load(os.path.join(MODEL_DIR, PRETRAINED_RED), device="auto")
model_red.policy.load_state_dict(pretrained_model.policy.state_dict(), strict=False)

# === Callbacks ===
callbacks = [
    EvalCallback(
        Env_Fase6_MapaGrande(),
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
print(f"🚀 Entrenando PPO rojo contra azul congelado en Fase 6 ({MODEL_NAME})...")
model_red.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model_red.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"✅ Modelo PPO rojo guardado como {MODEL_NAME}.zip")
