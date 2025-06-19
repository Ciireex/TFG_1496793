import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gym_strategy.envs.StrategyEnv_Fase0_v3 import StrategyEnv_Fase0_v2
from gym_strategy.utils.CustomCNN_Pro import CustomCNN_Pro

# === Dummy Wrapper para hacer que el equipo azul no actúe ===
class DummyBlueWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 1:
            return self.env.step(action)
        else:
            return self.env.step(0)  # Acción nula para azul

# === Configurar entorno y vectorización ===
def make_env():
    return DummyBlueWrapper(StrategyEnv_Fase0_v2())

env = DummyVecEnv([make_env])
env = VecMonitor(env)

# === Crear modelo PPO ===
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=2.5e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1,
    tensorboard_log="./logs/ppo_red/",
    policy_kwargs=dict(
        features_extractor_class=CustomCNN_Pro,
    ),
)

# === Callbacks ===
checkpoint_cb = CheckpointCallback(save_freq=50000, save_path="./models/", name_prefix="ppo_red")
eval_cb = EvalCallback(env, best_model_save_path="./models/", log_path="./logs/ppo_red/", eval_freq=10000)

# === Entrenar ===
model.learn(total_timesteps=300000, callback=[checkpoint_cb, eval_cb])
model.save("./models/final_model_red")
