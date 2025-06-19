import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from gym_strategy.envs.StrategyEnv_TransferMedium_1v1_Archers import StrategyEnv_TransferMedium_1v1_Archers
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

### === WRAPPER PARA ENTRENAR AL ROJO CONTRA UN AZUL FIJO === ###
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


### === FUNCIÓN DE ENTRENAMIENTO === ###
def train_red_against_fixed_blue(model_save_path, fixed_blue_model_path, log_dir):
    print(f"📥 Cargando modelo AZUL congelado desde: {fixed_blue_model_path}")

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    def make_env():
        env = StrategyEnv_TransferMedium_1v1_Archers()
        return FixedBlueWrapper(env, model_path=fixed_blue_model_path)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        policy_kwargs=dict(
            features_extractor_class=CustomCNN
        )
    )
    model.set_logger(new_logger)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path, name_prefix="model")
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, eval_freq=20000, deterministic=True, render=False)

    # Entrenar
    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(model_save_path, "final_model"))
    print("💾 Modelo ROJO final guardado en:", os.path.join(model_save_path, "final_model"))


if __name__ == "__main__":
    train_red_against_fixed_blue(
        model_save_path="./logs/transfer_red_vs_fixed_blue_fase3",
        fixed_blue_model_path="./logs/transfer_blue_vs_fixed_red_fase2/final_model",  # AZUL preentrenado
        log_dir="./logs/transfer_red_vs_fixed_blue_fase3"
    )
