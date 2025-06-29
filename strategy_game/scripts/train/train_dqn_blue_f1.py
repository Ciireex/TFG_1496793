import os
import sys
import random
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase1_Soldiers4x4
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CONFIGURACIÓN GENERAL ===
TOTAL_TIMESTEPS = 500_000
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../logs/dqn/blue_f1"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === CONFIGURACIÓN DE LA POLÍTICA ===
policy_kwargs = dict(
    features_extractor_class=EnhancedTacticalFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=384),
    net_arch=[256, 128]  # Para DQN se pasa una lista, no dict
)

# === ENVOLTORIO HEURÍSTICO PARA EL ROJO ===
class RedHeuristicWrapper(Env_Fase1_Soldiers4x4):
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
env = RedHeuristicWrapper()

# === MODELO DQN ===
model = DQN(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    gamma=0.99,
    seed=789,
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
        name_prefix="dqn_blue_f1"
    )
]

# === ENTRENAMIENTO ===
print("🚀 Entrenando modelo DQN azul F1 (contra heurística roja)...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# === GUARDADO FINAL ===
model.save(os.path.join(MODEL_DIR, "dqn_blue_f1"))
print("✅ Modelo DQN azul guardado como dqn_blue_f1.zip")
