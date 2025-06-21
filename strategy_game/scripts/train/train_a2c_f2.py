import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase2_Soldiers6x4_Obst
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# Configuración
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
CURRENT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../logs/a2c_blue_def_f2"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../models"))
MODEL_PATH_BLUE = os.path.join(MODEL_DIR, "a2c_blue_def_f1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Crear nuevo entorno (6x4 con obstáculos)
env_blue = make_vec_env(Env_Fase2_Soldiers6x4_Obst, n_envs=N_ENVS, seed=123)

# Red a transferir
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# Inicializar modelo en el nuevo entorno
model_blue = A2C(
    policy="CnnPolicy",
    env=env_blue,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=123,
    device="auto"
)

# Cargar pesos desde modelo anterior (transfer learning)
print("🧠 Transfer learning: cargando pesos desde Fase 1...")
pretrained = A2C.load(MODEL_PATH_BLUE)
model_blue.set_parameters(pretrained.get_parameters())
print("✅ Transferencia completada.")

# Callbacks
callbacks = [
    EvalCallback(
        Env_Fase2_Soldiers6x4_Obst(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="a2c_blue_def_f2"
    )
]

# Entrenamiento
print("🚀 Entrenando modelo A2C azul en Fase 2 (6x4 con obstáculos)...")
model_blue.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# Guardado
model_blue.save(os.path.join(MODEL_DIR, "a2c_blue_def_f2"))
print("✅ Modelo A2C azul guardado como a2c_blue_def_f2.zip")
