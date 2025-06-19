import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# === Rutas del proyecto ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Def import Env_Fase4_Archer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
MODEL_NAME = "ppo_blue_def_f3"
PRETRAINED_BLUE = "ppo_blue_def_f2"
RED_MODEL = "ppo_red_def_f2"
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/" + MODEL_NAME))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Wrapper personalizado con rojo congelado ===
class RedFrozenWrapper_Fase4(Env_Fase4_Archer):
    def __init__(self, red_model):
        super().__init__()
        self.red_model = red_model

    def step(self, action):
        if self.current_player == 0:
            obs, reward, terminated, truncated, info = super().step(action)
        else:
            obs = self._get_obs()
            action, _ = self.red_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

# === Cargar modelo rojo congelado ===
print("🔴 Cargando modelo PPO rojo:", RED_MODEL)
model_red = PPO.load(os.path.join(MODEL_DIR, RED_MODEL), device="auto")

# === Crear entorno Dummy con wrapper ===
def make_env():
    return RedFrozenWrapper_Fase4(model_red)

env = DummyVecEnv([make_env])

# === Cargar modelo azul preentrenado ===
print("🔵 Cargando modelo PPO azul:", PRETRAINED_BLUE)
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)
model_blue = PPO.load(os.path.join(MODEL_DIR, PRETRAINED_BLUE), env=env, device="auto")

# === Callbacks ===
callbacks = [
    EvalCallback(
        Env_Fase4_Archer(),
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
print(f"🚀 Entrenando PPO azul contra rojo congelado en Fase 4 ({MODEL_NAME})...")
model_blue.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model_blue.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"✅ Modelo PPO azul guardado como {MODEL_NAME}.zip")
