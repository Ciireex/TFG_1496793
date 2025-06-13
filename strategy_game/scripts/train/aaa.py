import os, sys, gymnasium as gym, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Rutas y entorno
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1_Archers import StrategyEnv_TransferSmall_1v1_Archers
from gym_strategy.utils.CustomCNN import CustomCNN

# Wrappers
class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
    def step(self, action):
        if self.env.current_player == 0:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs_tensor = torch.tensor(self.env._get_obs()).unsqueeze(0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(int(model_action))
            reward = 0.0
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = 0
        return obs, info

class FixedBlueWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
    def step(self, action):
        if self.env.current_player == 1:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs_tensor = torch.tensor(self.env._get_obs()).unsqueeze(0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(int(model_action))
            reward = 0.0
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = 1
        return obs, info

# === Función genérica para seguir entrenando ===
def continue_training(model_path, log_dir, env_wrapper):
    env_fn = lambda: env_wrapper(StrategyEnv_TransferSmall_1v1_Archers(), model_path)
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)
    model = PPO.load(model_path, env=env, custom_objects={"features_extractor_class": CustomCNN})
    model.set_env(env)
    print(f"🔁 Continuando entrenamiento desde {model_path} ...")
    model.learn(total_timesteps=1_000_000)
    model.save(model_path)
    print(f"✅ Entrenamiento extendido guardado en: {model_path}")

# === Continuar entrenamiento AZUL ===
continue_training(
    model_path="./logs/transfer_blue_vs_fixed_red/final_model",
    log_dir="./logs/transfer_blue_vs_fixed_red",
    env_wrapper=FixedRedWrapper
)

# === Continuar entrenamiento ROJO ===
continue_training(
    model_path="./logs/transfer_red_vs_fixed_blue/final_model",
    log_dir="./logs/transfer_red_vs_fixed_blue",
    env_wrapper=FixedBlueWrapper
)
