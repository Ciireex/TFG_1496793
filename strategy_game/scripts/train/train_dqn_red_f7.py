import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium import Wrapper

# === RUTAS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_MapaGrande
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 1_000_000
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/dqn/red_f7_v1"))
BLUE_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_blue_f7_v1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === POLICY KWARGS CNN ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[256, 128]
)

# === CARGAR MODELO AZUL CONGELADO (solo para inferencia) ===
print("Cargando modelo azul congelado...")
dummy_env = Env_Fase7_MapaGrande()
frozen_blue = DQN(
    policy="CnnPolicy",
    env=dummy_env,
    policy_kwargs=policy_kwargs,
    device="auto",
    verbose=0
)
data, params, _ = load_from_zip_file(BLUE_MODEL_PATH)
frozen_blue.policy.load_state_dict(params, strict=False)
print("✅ Modelo azul cargado para usar como oponente congelado.")

# === WRAPPER PARA INFERENCIA DEL AZUL ===
class BlueFrozenPolicyWrapper(Wrapper):
    def __init__(self, env, blue_model):
        super().__init__(env)
        self.blue_model = blue_model

    def step(self, action_red):
        if self.env.current_player == 1:
            obs, reward, done, trunc, info = self.env.step(action_red)
        else:
            obs_blue = self.env._get_obs()
            action_blue, _ = self.blue_model.predict(obs_blue, deterministic=True)
            obs, reward, done, trunc, info = self.env.step(action_blue)
        return obs, reward, done, trunc, info

# === CREAR ENTORNO CON WRAPPER ===
def make_env():
    return BlueFrozenPolicyWrapper(Env_Fase7_MapaGrande(), frozen_blue)

env = make_env()

# === CREAR MODELO DQN ROJO ===
print("🧠 Creando modelo DQN rojo vs azul congelado...")
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
        eval_env=make_env(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15_000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_red_f7_v1"
    )
]

# === ENTRENAMIENTO ===
print("Entrenando modelo DQN rojo en Fase 7 vs azul congelado...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "dqn_red_f7_v1"))
print("Modelo DQN rojo F7 guardado como dqn_red_f7_v1.zip")
