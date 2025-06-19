import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2Soldiers1Archer_6x4_Obs

# === Añadir path base ===


# === Clase Wrapper para fijar el BLUE ===
class FrozenBlueWrapper(StrategyEnv_2Soldiers1Archer_6x4_Obs):
    def __init__(self, blue_model):
        super().__init__()
        self.model_blue = blue_model

    def step(self, action):
        if self.current_player == 0:
            # El azul usa su política congelada
            obs = self._get_obs()
            action, _ = self.model_blue.predict(obs, deterministic=True)
        return super().step(action)

# === Rutas generales ===
CURRENT_DIR = os.path.dirname(__file__)
BLUE_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_2soldiers1archer/ppo_2soldiers1archer_final.zip"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models/ppo_red_vs_frozenblue_2s1a"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../logs/ppo_red_vs_frozenblue_2s1a"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Cargar el modelo azul congelado ===
print("🧊 Cargando modelo BLUE congelado...")
model_blue = PPO.load(BLUE_MODEL_PATH)

# === Entorno env donde el BLUE está congelado y RED aprende ===
def make_env():
    return FrozenBlueWrapper(model_blue)
env = DummyVecEnv([make_env])

# === Definir política y modelo para RED ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

model_red = PPO(
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
    seed=42,
    device="auto"
)

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
        name_prefix="ppo_red_vs_frozenblue"
    )
]

# === Entrenar al modelo RED ===
print("🚀 Entrenando al RED contra BLUE congelado...")
model_red.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)
model_red.save(os.path.join(MODEL_DIR, "ppo_red_vs_frozenblue_final.zip"))
print("✅ Modelo RED guardado como ppo_red_vs_frozenblue_final.zip")
