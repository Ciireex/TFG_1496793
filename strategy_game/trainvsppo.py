import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2
import gymnasium as gym

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
        print(f"[RESET] Episodio #{self.episode_count} (Equipo {self.controlled_team})")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not terminated and not truncated and self.env.current_player != self.controlled_team:
            action, _ = self.opponent_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, info

def train_self_play():
    model_blue_path = "models/selfplay_ppo_blue_cycle0.zip"
    model_red_path = "models/selfplay_ppo_red_cycle0.zip"

    model_blue = PPO.load("models/ppo_vs_heuristic_v3", device="cpu")
    model_red = PPO.load("models/ppo_rojo_vs_heuristica_azul.zip", device="cpu")

    model_blue.save(model_blue_path)
    model_red.save(model_red_path)

    n_cycles = 10
    timesteps_per_cycle = 100_000

    for cycle in range(1, n_cycles + 1):
        print(f"\n🔁 CICLO {cycle}")

        # Entrena BLUE contra RED actual
        opponent_red = PPO.load(model_red_path, device="cpu")
        env_blue = DummyVecEnv([lambda: SelfPlayEnvWrapper(StrategyEnvPPOA2C2(), controlled_team=0, opponent_model=opponent_red)])
        model_blue.set_env(env_blue)
        model_blue.learn(total_timesteps=timesteps_per_cycle, reset_num_timesteps=False)
        model_blue_path = f"models/selfplay_ppo_blue_cycle{cycle}.zip"
        model_blue.save(model_blue_path)
        print(f"✅ BLUE guardado en {model_blue_path}")

        # Entrena RED contra BLUE actualizado
        opponent_blue = PPO.load(model_blue_path, device="cpu")
        env_red = DummyVecEnv([lambda: SelfPlayEnvWrapper(StrategyEnvPPOA2C2(), controlled_team=1, opponent_model=opponent_blue)])
        model_red.set_env(env_red)
        model_red.learn(total_timesteps=timesteps_per_cycle, reset_num_timesteps=False)
        model_red_path = f"models/selfplay_ppo_red_cycle{cycle}.zip"
        model_red.save(model_red_path)
        print(f"✅ RED guardado en {model_red_path}")

    print("\n✅ ENTRENAMIENTO SELF-PLAY COMPLETADO")

if __name__ == "__main__":
    train_self_play()
