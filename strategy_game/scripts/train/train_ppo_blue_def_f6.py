import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Añadir ruta del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Def import Env_Fase6_MapaGrande
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
MODEL_NAME = "ppo_blue_def_f6"
PRETRAINED_BLUE = "ppo_blue_def_f4"
RED_MODEL = "ppo_red_def_f4"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/" + MODEL_NAME))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class RedFrozenWrapper_Fase6(Env_Fase6_MapaGrande):
    def __init__(self, red_model):
        super().__init__()
        self.red_model = red_model

    def step(self, action):
        if self.current_player == 0:
            # Azul (entrena)
            obs, reward, terminated, truncated, info = super().step(action)
        else:
            # Rojo (congelado) – recortamos observación antes del predict
            obs = self._get_obs()
            obs_resized = obs[:, :6, :4]  # Recorta a (21, 6, 4) para el modelo rojo
            action, _ = self.red_model.predict(obs_resized, deterministic=True)
            obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

# === Cargar modelo rojo congelado ===
print("🔴 Cargando modelo PPO rojo:", RED_MODEL)
model_red = PPO.load(os.path.join(MODEL_DIR, RED_MODEL), device="auto")

# === Crear entorno Dummy con wrapper ===
def make_env():
    return RedFrozenWrapper_Fase6(model_red)

env = DummyVecEnv([make_env])

# === Transfer learning desde modelo anterior ===
print("🔵 Aplicando transfer learning desde:", PRETRAINED_BLUE)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# Crear modelo nuevo en entorno grande
model_blue = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
    device="auto"
)

# Cargar pesos del modelo anterior entrenado en 6x4
pretrained_model = PPO.load(os.path.join(MODEL_DIR, PRETRAINED_BLUE), device="auto")
model_blue.policy.load_state_dict(pretrained_model.policy.state_dict(), strict=False)

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
print(f"🚀 Entrenando PPO azul contra rojo congelado en Fase 6 ({MODEL_NAME})...")
model_blue.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model_blue.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"✅ Modelo PPO azul guardado como {MODEL_NAME}.zip")
