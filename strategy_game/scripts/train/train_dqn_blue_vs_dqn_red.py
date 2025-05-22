import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("PYTHONPATH añadido:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnv import StrategyEnv
import gymnasium as gym

# === CONFIGURACIÓN ===
save_path = "models"
os.makedirs(save_path, exist_ok=True)
log_dir = "./logs/selfplay_ppoblue_vs_ppored/"
total_cycles = 10
timesteps_per_cycle = 100_000

# === Máscara de acciones válidas ===
def mask_fn(env):
    return env.valid_action_mask()

# === Wrapper para Self-Play entre dos agentes PPO ===
class SelfPlayEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team, opponent_model):
        super().__init__(base_env)
        self.controlled_team = controlled_team
        self.opponent_model = opponent_model
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        self.episode_count += 1
        print(f"[RESET] Episodio #{self.episode_count}")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def valid_action_mask(self):
        return self.env.valid_action_mask()

# === ENTRENAMIENTO POR CICLOS ===
blue_model_path = os.path.join(save_path, "ppoblue_vs_heuristic_curriculum_maskable_v1.zip")
red_model_path = os.path.join(save_path, "ppored_vs_heuristic_curriculum_maskable_v1.zip")

for cycle in range(1, total_cycles + 1):
    print(f"\n🔁 Ciclo {cycle}/10 - Entrenando equipo azul contra rojo...")
    opponent_red = MaskablePPO.load(red_model_path, device="cpu")

    env_blue = DummyVecEnv([lambda: ActionMasker(
        SelfPlayEnvWrapper(StrategyEnv(use_obstacles=True), controlled_team=0, opponent_model=opponent_red),
        mask_fn
    )])
    model_blue = MaskablePPO("MlpPolicy", env_blue, verbose=1, device="cpu", tensorboard_log=log_dir)
    model_blue.learn(total_timesteps=timesteps_per_cycle)
    blue_model_path = os.path.join(save_path, f"ppoblue_vs_ppored_cycle{cycle}.zip")
    model_blue.save(blue_model_path)

    print(f"\n🔁 Ciclo {cycle}/10 - Entrenando equipo rojo contra azul...")
    opponent_blue = MaskablePPO.load(blue_model_path, device="cpu")

    env_red = DummyVecEnv([lambda: ActionMasker(
        SelfPlayEnvWrapper(StrategyEnv(use_obstacles=True), controlled_team=1, opponent_model=opponent_blue),
        mask_fn
    )])
    model_red = MaskablePPO("MlpPolicy", env_red, verbose=1, device="cpu", tensorboard_log=log_dir)
    model_red.learn(total_timesteps=timesteps_per_cycle)
    red_model_path = os.path.join(save_path, f"ppored_vs_ppoblue_cycle{cycle}.zip")
    model_red.save(red_model_path)

print("\n✅ ENTRENAMIENTO ENTRE AMBOS AGENTES COMPLETADO ✅")
