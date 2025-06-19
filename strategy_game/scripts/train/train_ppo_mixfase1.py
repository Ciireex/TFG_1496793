import os, sys, gymnasium as gym, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# Añadir ruta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1_Archers import StrategyEnv_TransferSmall_1v1_Archers
from gym_strategy.utils.CustomCNN_Pro import CustomCNN

### === WRAPPERS === ###

class FixedRedWrapper(gym.Wrapper):
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = PPO.load(model_path, custom_objects={"features_extractor_class": CustomCNN})

    def step(self, action):
        if self.env.current_player == 0:  # AZUL aprende
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:  # ROJO fijo
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
        if self.env.current_player == 1:  # ROJO aprende
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:  # AZUL fijo
            obs_tensor = torch.tensor(self.env._get_obs()).unsqueeze(0)
            model_action, _ = self.model.predict(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(int(model_action))
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.current_player = 1
        return obs, info

### === ENTRENAMIENTO === ###

def train_model(env_fn, log_dir, model_name, prev_model_path=None):
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "/best_model",
                                 log_path=log_dir + "/eval", eval_freq=5000, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir + "/checkpoints", name_prefix=model_name)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if prev_model_path:
        model = PPO.load(prev_model_path, env=env, custom_objects={"features_extractor_class": CustomCNN})
        model.set_env(env)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=500_000, callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(log_dir, "final_model"))
    print(f"✅ Entrenamiento completado: {model_name}")
    return os.path.join(log_dir, "final_model")

### === EJECUCIÓN SECUENCIAL === ###

# Ruta a modelo base del rojo entrenado en 2v2 sin arqueros
model_red_base_path = "./logs/ppo_red_vs_random/final_model"

# 1. AZUL contra ROJO fijo
print("\n=== ENTRENANDO AZUL (4v4) VS ROJO FIJO (2v2) ===")
model_blue_archers_path = train_model(
    env_fn=lambda: FixedRedWrapper(StrategyEnv_TransferSmall_1v1_Archers(), model_red_base_path),
    log_dir="./logs/transfer_blue_vs_fixed_red",
    model_name="ppo_blue_archers"
)

# 2. ROJO contra AZUL fijo (recién entrenado)
print("\n=== ENTRENANDO ROJO (4v4) VS AZUL FIJO (4v4) ===")
train_model(
    env_fn=lambda: FixedBlueWrapper(StrategyEnv_TransferSmall_1v1_Archers(), model_blue_archers_path),
    log_dir="./logs/transfer_red_vs_fixed_blue",
    model_name="ppo_red_archers"
)
