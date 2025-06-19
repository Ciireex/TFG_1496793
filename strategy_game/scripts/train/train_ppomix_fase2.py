import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import numpy as np
import torch
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from gym_strategy.envs.StrategyEnv_Fase2 import StrategyEnv_Fase2
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

### === WRAPPER ROJO ALEATORIO (MUEVE Y ATACA) === ###
class RandomRedWrapper(gym.Wrapper):
    def step(self, action):
        if self.env.current_player == 0:  # Azul
            return self.env.step(action)
        else:
            unit = self.env._get_active_unit()
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            # Si hay enemigo en rango, atacar
            for d_idx, (dx, dy) in enumerate(dirs):
                for dist in range(getattr(unit, "min_range", 1), unit.attack_range + 1):
                    tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
                    if not self.env._valid_coord((tx, ty)):
                        break
                    for enemy in self.env.units:
                        if enemy.team != unit.team and enemy.is_alive() and enemy.position == (tx, ty):
                            return self.env.step(d_idx + 1)  # Dirección de ataque

            # Si no, moverse aleatorio
            return self.env.step(random.randint(0, 4))  # 0 = quieto, 1–4 = direcciones

### === ENTRENAMIENTO AZUL VS ROJO ALEATORIO === ###
def train_blue_vs_random_red(model_save_path, log_dir):
    print("🎯 Entrenando AZUL contra ROJO aleatorio")
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    def make_env():
        env = StrategyEnv_Fase2(use_obstacles=False)
        return RandomRedWrapper(env)

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

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path, name_prefix="model")
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, eval_freq=20000, deterministic=True, render=False)

    model.learn(total_timesteps=300_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(model_save_path, "final_model"))
    print("💾 Modelo AZUL guardado en:", os.path.join(model_save_path, "final_model"))

if __name__ == "__main__":
    train_blue_vs_random_red(
        model_save_path="./logs/fase2_blue_vs_random_red",
        log_dir="./logs/fase2_blue_vs_random_red"
    )
