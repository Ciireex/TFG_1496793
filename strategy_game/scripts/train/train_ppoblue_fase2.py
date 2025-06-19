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
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1_Archers import StrategyEnv_TransferSmall_1v1_Archers
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

### === WRAPPER PARA USAR MODELO ROJO FIJO ENTRENADO EN ENTORNO 6x4 === ###
class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        # Cargar modelo PPO entrenado en entorno 6x4
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})
        self.model.policy.eval()
        # Crear entorno de 6x4 para usar como dummy al hacer predicciones (estructura compatible con el modelo)
        self.dummy_env = StrategyEnv_TransferSmall_1v1_Archers()

    def step(self, action):
        if self.env.current_player == 0:  # Azul aprende
            return self.env.step(action)
        else:  # Rojo fijo
            obs = self.dummy_env._get_obs()  # usar obs 6x4 compatible
            obs_tensor = np.expand_dims(obs, axis=0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            return self.env.step(int(model_action))

    def reset(self, **kwargs):
        self.dummy_env.reset()
        return self.env.reset(**kwargs)

### === FUNCION DE ENTRENAMIENTO === ###
def train_transfer(model_base_path, log_dir, model_save_path, env_wrapper_class):
    print(f"📥 Cargando pesos desde: {model_base_path}")

    # Logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Crear entorno de entrenamiento con wrapper
    def make_env():
        env = StrategyEnv_TransferMedium_1v1_Archers()  # entorno 8x6
        return env_wrapper_class(env, model_base_path)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # Crear modelo nuevo (con CustomCNN)
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

    # Transferencia parcial de CNN del modelo antiguo
    old_model = PPO.load(model_base_path, custom_objects={"features_extractor_class": CustomCNN})
    with torch.no_grad():
        model.policy.features_extractor.cnn.load_state_dict(
            old_model.policy.features_extractor.cnn.state_dict()
        )
    print("✅ Transferencia parcial de pesos completada (solo CNN).")

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path, name_prefix="model")
    eval_callback = EvalCallback(env, best_model_save_path=model_save_path, eval_freq=20000, deterministic=True, render=False)

    # Entrenar
    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(model_save_path, "final_model"))
    print("💾 Modelo final guardado en:", os.path.join(model_save_path, "final_model"))

if __name__ == "__main__":
    train_transfer(
        model_base_path="./logs/transfer_blue_vs_fixed_red/final_model",  # modelo entrenado en 6x4
        log_dir="./logs/transfer_blue_vs_fixed_red_fase2",                # logs de entrenamiento actual
        model_save_path="./logs/transfer_blue_vs_fixed_red_fase2",        # destino de checkpoints y modelo final
        env_wrapper_class=FixedRedWrapper
    )
