import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from gym_strategy.envs.StrategyEnv_CastleControl import StrategyEnv_CastleControl
from gym_strategy.utils.CustomCNN import CustomCNN

# Detectar automáticamente si hay GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

### === WRAPPERS === ###
class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model.policy.eval()

    def step(self, action):
        if self.env.current_player == 0:  # AZUL aprende
            return self.env.step(action)
        else:  # ROJO fijo
            obs = self.env._get_obs()
            obs_tensor = np.expand_dims(obs, axis=0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            return self.env.step(int(model_action))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FixedBlueWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model.policy.eval()

    def step(self, action):
        if self.env.current_player == 1:  # ROJO aprende
            return self.env.step(action)
        else:  # AZUL fijo
            obs = self.env._get_obs()
            obs_tensor = np.expand_dims(obs, axis=0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            return self.env.step(int(model_action))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


### === ENTRENAMIENTO === ###
def train_blue_against_fixed_red(model_save_path, fixed_red_model_path, prev_model_path, log_dir):
    print(f"Entrenando AZUL contra ROJO congelado: {fixed_red_model_path}")
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    def make_env():
        env = StrategyEnv_CastleControl()
        return FixedRedWrapper(env, model_path=fixed_red_model_path)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    model = PPO.load(
        prev_model_path,
        env=env,
        device=device,
        custom_objects={"features_extractor_class": CustomCNN}
    )
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path, name_prefix="model")
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, eval_freq=20000, deterministic=True, render=False)

    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(model_save_path, "final_model"))
    print("Modelo AZUL final guardado en:", os.path.join(model_save_path, "final_model"))


def train_red_against_fixed_blue(model_save_path, fixed_blue_model_path, prev_model_path, log_dir):
    print(f"Entrenando ROJO contra AZUL congelado: {fixed_blue_model_path}")
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    def make_env():
        env = StrategyEnv_CastleControl()
        return FixedBlueWrapper(env, model_path=fixed_blue_model_path)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    model = PPO.load(
        prev_model_path,
        env=env,
        device=device,
        custom_objects={"features_extractor_class": CustomCNN}
    )
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path, name_prefix="model")
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, eval_freq=20000, deterministic=True, render=False)

    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(model_save_path, "final_model"))
    print("Modelo ROJO final guardado en:", os.path.join(model_save_path, "final_model"))


if __name__ == "__main__":
    # Fase 3: entrenar AZUL vs ROJO congelado, usando modelo anterior como base
    train_blue_against_fixed_red(
        model_save_path="./logs/castle_blue_vs_fixed_red_fase3",
        fixed_red_model_path="./logs/castle_red_vs_fixed_blue_fase2/final_model",
        prev_model_path="./logs/castle_blue_vs_fixed_red_fase2/final_model",
        log_dir="./logs/castle_blue_vs_fixed_red_fase3"
    )

    # Fase 3: entrenar ROJO vs AZUL recién entrenado, usando modelo anterior como base
    train_red_against_fixed_blue(
        model_save_path="./logs/castle_red_vs_fixed_blue_fase3",
        fixed_blue_model_path="./logs/castle_blue_vs_fixed_red_fase3/final_model",
        prev_model_path="./logs/castle_red_vs_fixed_blue_fase2/final_model",
        log_dir="./logs/castle_red_vs_fixed_blue_fase3"
    )