import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv
from gym_strategy.utils.HeuristicPolicy import HeuristicPolicy

# Función para obtener la máscara de acciones válidas
def mask_fn(env):
    return env.valid_action_mask()

# Wrapper para controlar un equipo y dejar el otro con heurística
class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.heuristic = HeuristicPolicy(base_env)
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        print(f"[RESET] Episodio #{self.episode_count}")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action = self.heuristic.get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def valid_action_mask(self):
        return self.env.valid_action_mask()

# === ENTRENAMIENTO MaskablePPO Azul v3 VS Heurística Roja ===
print("🏁 Entrenando MaskablePPO v3 (azul) vs heurística v2 (rojo) con obstáculos (1M pasos)...")

env = DummyVecEnv([lambda: ActionMasker(
    DualTeamEnvWrapper(StrategyEnv(use_obstacles=True), controlled_team=0),
    mask_fn
)])

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/maskableppoblue_vs_heuristic_v3/",
    device="cpu"
)

model.learn(
    total_timesteps=1_000_000,
    tb_log_name="MaskablePPO_vs_Heuristic_v3",
    reset_num_timesteps=True
)

# Guardar modelo final
model.save("models/maskableppoblue_vs_heuristic_v3")

print("\n✅ ENTRENAMIENTO COMPLETADO Y GUARDADO: maskableppoblue_vs_heuristic_v3 ✅")
