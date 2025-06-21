import os
import sys
import random
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Def import Env_Fase2_Soldiers6x4_Obst
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/a2c_blue_def_f1_continue"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_blue_def_f1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === ENVOLTORIO HEURÍSTICO PARA ROJO ===
class RedHeuristicWrapper(Env_Fase2_Soldiers6x4_Obst):
    def step(self, action):
        if self.current_player == 1:  # rojo = heurístico
            unit = self._get_active_unit()
            if not unit:
                return super().step(0)

            if self.phase == "attack":
                x, y = unit.position
                for dir_idx, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    tx, ty = x + dx, y + dy
                    if not self._valid_coord((tx, ty)):
                        continue
                    for enemy in [u for u in self.units if u.team != unit.team and u.is_alive()]:
                        if enemy.position == (tx, ty):
                            return Env_Fase2_Soldiers6x4_Obst.step(self, dir_idx + 1)
                return Env_Fase2_Soldiers6x4_Obst.step(self, 0)
            else:
                x, y = unit.position
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return Env_Fase2_Soldiers6x4_Obst.step(self, directions.index((dx, dy)) + 1)
                return Env_Fase2_Soldiers6x4_Obst.step(self, 0)
        else:
            return Env_Fase2_Soldiers6x4_Obst.step(self, action)

# === ENTORNO DE ENTRENAMIENTO ===
env = make_vec_env(lambda: RedHeuristicWrapper(), n_envs=N_ENVS, seed=789)

# === CARGAR MODELO PREVIO Y CONTINUAR ENTRENANDO ===
model = A2C.load(PREV_MODEL_PATH, env=env, device="auto")
model.tensorboard_log = LOG_DIR
model.set_parameters(PREV_MODEL_PATH, exact_match=True)

# === AJUSTAR PARÁMETROS DE ENTRENAMIENTO ===
model.n_steps = 32  # mejora temporal para A2C
model.learning_rate = 7e-4
model.ent_coef = 0.01
model.gamma = 0.99
model.gae_lambda = 1.0
model.max_grad_norm = 0.5

# === CALLBACKS ===
callbacks = [
    EvalCallback(
        RedHeuristicWrapper(),
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="a2c_blue_def_f1_continue"
    )
]

# === INICIO ENTRENAMIENTO ===
print("🚀 Continuando entrenamiento A2C azul contra heurístico...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO DEL MODELO FINAL ===
model.save(os.path.join(MODEL_DIR, "a2c_blue_def_f1_continue"))
print("✅ Modelo A2C azul guardado como a2c_blue_def_f1_continue.zip")
