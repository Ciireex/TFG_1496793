import os
import sys
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/ppo_blue_def_f0"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=dict(pi=[256, 128], vf=[256, 128])
)

# === ENVOLTORIO HEURÍSTICO PARA ROJO ===
class RedHeuristicWrapper(Env_Fase1_Soldiers4x4):
    def step(self, action):
        if self.current_player == 1:  # IA heurística para rojo
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
                            return Env_Fase1_Soldiers4x4.step(self, dir_idx + 1)
                return Env_Fase1_Soldiers4x4.step(self, 0)
            else:
                x, y = unit.position
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return Env_Fase1_Soldiers4x4.step(self, directions.index((dx, dy)) + 1)
                return Env_Fase1_Soldiers4x4.step(self, 0)
        else:
            return Env_Fase1_Soldiers4x4.step(self, action)

# === ENTORNO DE ENTRENAMIENTO ===
env = make_vec_env(lambda: RedHeuristicWrapper(), n_envs=N_ENVS, seed=123)

# === ENTRENAMIENTO PPO ===
model = PPO(
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
    seed=123,
    device="auto"
)

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
        name_prefix="ppo_blue_def_f0"
    )
]

# === INICIO ENTRENAMIENTO ===
print("🚀 Entrenando modelo PPO azul en Fase 1 (Def)...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO DEL MODELO ===
model.save(os.path.join(MODEL_DIR, "ppo_blue_def_f0"))
print("✅ Modelo PPO azul guardado como ppo_blue_def_f0.zip")
