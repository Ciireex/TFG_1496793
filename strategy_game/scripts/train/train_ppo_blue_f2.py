import os
import sys
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_3v3Soldiers6x4_Obs
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIG ===
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LOG_DIR = "../logs/ppo_blue_vs_heuristicred_vf2/"
MODEL_DIR = "../models/ppo_blue_vs_heuristicred/"
PRETRAINED_PATH = os.path.join(MODEL_DIR, "ppo_blue_vf1.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[dict(pi=[256, 128], vf=[256, 128])]
)

# === HEURÍSTICA ROJA ===
class RedHeuristicWrapper(StrategyEnv_3v3Soldiers6x4_Obs):
    def step(self, action):
        if self.current_player == 1:
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

# === ENV ===
env = make_vec_env(lambda: RedHeuristicWrapper(), n_envs=N_ENVS, seed=789)

# === TRANSFERENCIA DE PESOS ===
print("🧠 Cargando modelo anterior ppo_blue_vf1...")
model_old = PPO.load(PRETRAINED_PATH)

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
    seed=789,
    device="auto"
)

model.set_parameters(model_old.get_parameters())
print("✅ Pesos cargados desde ppo_blue_vf1.zip")

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
        name_prefix="ppo_blue_vf2"
    )
]

# === ENTRENAMIENTO ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
model.save(os.path.join(MODEL_DIR, "ppo_blue_vf2"))
print("✅ Modelo azul entrenado guardado como ppo_blue_vf2.zip")
