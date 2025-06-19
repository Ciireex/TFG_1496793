import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN GENERAL ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4

LOG_DIR_1 = "../logs/ppo_blue_vs_heuristicred/"
MODEL_DIR_1 = "../models/ppo_blue_vs_heuristicred/"
LOG_DIR_2 = "../logs/ppo_red_vs_frozenblue/"
MODEL_DIR_2 = "../models/ppo_red_vs_frozenblue/"

os.makedirs(LOG_DIR_1, exist_ok=True)
os.makedirs(MODEL_DIR_1, exist_ok=True)
os.makedirs(LOG_DIR_2, exist_ok=True)
os.makedirs(MODEL_DIR_2, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === FASE 1: Entrenar AZUL contra rojo heurístico ===
print("🔵 FASE 1: Entrenando AZUL contra ROJO heurístico...")

class RedHeuristicWrapper(StrategyEnv_2v2Soldiers4x4):
    def step(self, action):
        if self.current_player == 1:  # rojo
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
                                return super().step(dir_idx + 1)
                return super().step(0)
            else:
                x, y = active_unit.position
                directions = [(-1,0), (1,0), (0,-1), (0,1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self._valid_move((nx, ny)):
                        return super().step(directions.index((dx, dy)) + 1)
                return super().step(0)
        else:
            return super().step(action)

env1 = make_vec_env(lambda: RedHeuristicWrapper(), n_envs=N_ENVS, seed=123)

model1 = PPO(
    policy="CnnPolicy",
    env=env1,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR_1,
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

callback1 = [
    EvalCallback(
        RedHeuristicWrapper(),
        best_model_save_path=MODEL_DIR_1,
        log_path=LOG_DIR_1,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR_1,
        name_prefix="ppo_blue"
    )
]

model1.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback1, progress_bar=True)
model1.save(os.path.join(MODEL_DIR_1, "ppo_blue_vf0"))
print("✅ FASE 1 completada: Modelo azul guardado como ppo_blue_vf0")

# === FASE 2: Entrenar ROJO contra azul congelado ===
print("🔴 FASE 2: Entrenando ROJO contra AZUL congelado...")

class FrozenBlueWrapper(StrategyEnv_2v2Soldiers4x4):
    def __init__(self, blue_model_path):
        super().__init__()
        self.blue_model = PPO.load(blue_model_path)

    def step(self, action):
        if self.current_player == 0:  # azul congelado
            obs = self._get_obs()
            frozen_action, _ = self.blue_model.predict(obs, deterministic=True)
            return super().step(frozen_action)
        else:
            return super().step(action)

blue_model_path = os.path.join(MODEL_DIR_1, "ppo_blue_vf0.zip")

env2 = make_vec_env(lambda: FrozenBlueWrapper(blue_model_path), n_envs=N_ENVS, seed=456)

model2 = PPO(
    policy="CnnPolicy",
    env=env2,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR_2,
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    seed=456,
    device="auto"
)

callback2 = [
    EvalCallback(
        FrozenBlueWrapper(blue_model_path),
        best_model_save_path=MODEL_DIR_2,
        log_path=LOG_DIR_2,
        eval_freq=15000,
        deterministic=True,
        render=False
    ),
    CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR_2,
        name_prefix="ppo_red"
    )
]

model2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback2, progress_bar=True)
model2.save(os.path.join(MODEL_DIR_2, "ppo_red_vf0"))
print("✅ FASE 2 completada: Modelo rojo guardado como ppo_red_vf0")
