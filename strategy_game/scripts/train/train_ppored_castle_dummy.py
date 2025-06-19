import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

# === Dummy Heuristic que siempre pasa ===
class DummyHeuristic:
    def __init__(self, env):
        self.env = env
    def get_action(self, obs):
        return 0  # Acción 0 = pasar

# === Wrapper PPO (rojo) vs Dummy (azul) ===
class DualTeamDummyWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team=1):  # ← PPO controla ROJO
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.dummy = DummyHeuristic(base_env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action = self.dummy.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            d_action = self.dummy.get_action(obs)
            obs, _, terminated, truncated, info = self.env.step(d_action)
        return obs, reward, terminated, truncated, info

# === Crear entorno ===
def make_env():
    def _init():
        base_env = StrategyEnv_Castle(use_obstacles=True, obstacle_count=10)
        return DualTeamDummyWrapper(base_env, controlled_team=1)
    return _init

env = DummyVecEnv([make_env()])

# === Callback para guardar modelos ===
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/",
    name_prefix="ppo_red_vs_dummy"
)

# === Modelo PPO para el equipo rojo ===
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./logs/",
    device="cpu"
)

# === Entrenamiento ===
model.learn(total_timesteps=500_000, callback=checkpoint_callback)
model.save("models/ppo_red_vs_dummy")
