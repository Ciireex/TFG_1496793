import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.wrappers import FlattenObservation

from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# === CONFIGURACIÓN ===
N_EPISODES = 100
USE_OBSTACLES = False

MODELS = {
    "ppo_blue": ("models/ppoblue_vs_heuristic_v1", PPO, StrategyEnv, 0),
    "ppo_red": ("models/ppored_vs_heuristic_v1", PPO, StrategyEnv, 1),
    "a2c_blue": ("models/a2cblue_vs_heuristic_v1", A2C, StrategyEnv, 0),
    "a2c_red": ("models/a2cred_vs_heuristic_v1", A2C, StrategyEnv, 1),
    "dqn_blue": ("models/dqn_blue_vs_heuristic_v1", DQN, StrategyEnvPPOA2C2, 0),
    "dqn_red": ("models/dqn_red_vs_heuristic_v1", DQN, StrategyEnvPPOA2C2, 1),
    "maskableppo_blue": ("models/maskableppoblue_vs_heuristic_v2", MaskablePPO, StrategyEnv, 0),
    "maskableppo_red": ("models/maskableppored_vs_heuristic_v2", MaskablePPO, StrategyEnv, 1),
}

def mask_fn(env):
    return env.valid_action_mask()

# === EVALUACIÓN ===
for name, (path, cls, env_cls, team_id) in MODELS.items():
    print(f"\n🎮 Evaluando {name} vs IA heurística...")
    model = cls.load(path, device="cpu")
    is_maskable = issubclass(cls, MaskablePPO)
    is_dqn = issubclass(cls, DQN)

    wins, losses, draws = 0, 0, 0

    for _ in range(N_EPISODES):
        # Crear entorno según el tipo
        if env_cls == StrategyEnv:
            env = env_cls(use_obstacles=USE_OBSTACLES)
        else:
            env = env_cls()

        heuristic = HeuristicPolicy(env)

        if is_maskable:
            env = ActionMasker(env, mask_fn)
        if is_dqn:
            env = FlattenObservation(env)

        obs, _ = env.reset()
        done = False

        while not done:
            current_team = env.env.current_player if hasattr(env, "env") else env.current_player
            if current_team == team_id:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = heuristic.get_action(obs)
            obs, _, done, _, _ = env.step(action)

        # Resultado
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        alive = [u.team for u in base_env.units if u.is_alive()]

        if len(set(alive)) == 1:
            winner = alive[0]
            if winner == team_id:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    print(f"📊 Resultado {name}: {wins}🟦 victorias / {losses}🟥 derrotas / {draws}🤝 empates")
