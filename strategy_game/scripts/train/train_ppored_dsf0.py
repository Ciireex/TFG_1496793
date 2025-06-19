# train_ppored_vs_heuristicblue.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor
import random

# === Heurística azul: se mueve al azar y ataca si puede ===
class BlueHeuristicWrapper(StrategyEnv_2v2Soldiers4x4):
    def step(self, action):
        if self.current_player == 0:  # Azul: heurística
            active_unit = self._get_active_unit()
            if not active_unit:
                return super().step(0)

            if self.phase == "attack":
                x, y = active_unit.position
                for dir_idx, (dx, dy) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    for dist in range(1, active_unit.attack_range + 1):
                        tx, ty = x + dx * dist, y + dy * dist
                        if not self._valid_coord((tx, ty)):
                            break
                        for enemy in [u for u in self.units if u.team != active_unit.team and u.is_alive()]:
                            if enemy.position == (tx, ty):
                                return super().step(dir_idx + 1)  # ataca

                return super().step(0)  # no enemigo a rango → pasa

            else:  # fase de movimiento
                x, y = active_unit.position
                directions = [(-1,0), (1,0), (0,-1), (0,1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return super().step(directions.index((dx, dy)) + 1)

                return super().step(0)  # no hay movimiento válido

        else:
            return super().step(action)  # Rojo (PPO)

# === Configuración ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
SEED = 44
LOG_DIR = "./logs/ppo_red_vs_heuristicblue/"
MODEL_DIR = "./models/ppo_red_vs_heuristicblue/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Política personalizada ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === Crear entornos vectorizados ===
env = make_vec_env(
    lambda: BlueHeuristicWrapper(),
    n_envs=N_ENVS,
    seed=SEED
)

# === Callbacks ===
eval_callback = EvalCallback(
    BlueHeuristicWrapper(),
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=15_000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_red"
)

# === Entrenamiento ===
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
    seed=SEED,
    device="auto"
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# === Guardar modelo final ===
model.save(os.path.join(MODEL_DIR, "ppo_red_f0_v3"))
print("✅ Entrenamiento de PPO rojo contra azul heurístico completado.")
