import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_Castle import StrategyEnv_Castle
from gym_strategy.utils.HeuristicCastle import HeuristicCastle
from gym_strategy.utils.CustomCNN import CustomCNN

# --- Wrapper contra heurística ---
class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, env, controlled_team=0, opponent_policy=None):
        super().__init__(env)
        self.controlled_team = controlled_team
        self.opponent_policy = opponent_policy or HeuristicCastle(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            obs, _, terminated, truncated, _ = self.env.step(self.opponent_policy.get_action(obs))
            if terminated or truncated:
                break
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            obs, reward, terminated, truncated, info = self.env.step(self.opponent_policy.get_action(obs))
        return obs, reward, terminated, truncated, info

# --- Crear entorno env wrapped ---
def make_env(use_obstacles):
    base_env = StrategyEnv_Castle(use_obstacles=use_obstacles)
    return DualTeamEnvWrapper(base_env, controlled_team=0)

# --- Configuración de la CNN ---
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

# --- Entrenamiento PPO en dos fases ---
if __name__ == "__main__":
    env1 = DummyVecEnv([lambda: make_env(use_obstacles=False)])
    env2 = DummyVecEnv([lambda: make_env(use_obstacles=True)])

    model = PPO(
        policy="CnnPolicy",
        env=env1,
        verbose=1,
        tensorboard_log="./ppo_castle_tensorboard/",
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("🚀 Fase 1: Entrenamiento sin obstáculos (100k)")
    model.learn(total_timesteps=100_000)

    print("🚀 Fase 2: Entrenamiento con obstáculos (300k)")
    model.set_env(env2)
    model.learn(total_timesteps=300_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_castle_v2_mix")
    print("✅ Modelo guardado en: models/ppo_castle_v2_mix")
