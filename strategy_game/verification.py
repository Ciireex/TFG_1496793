import os
import gymnasium as gym
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnvSparteMinimal import StrategyEnv5x5Detailed

class DualTeamEnvWrapper(gym.Wrapper):
    def __init__(self, base_env, controlled_team):
        super().__init__(base_env)
        self.controlled_team = controlled_team

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while self.env.current_player != self.controlled_team:
            _, _, done, _, _ = self.env.step(0)
        return obs, info

    def step(self, action):
        if self.env.current_player == self.controlled_team:
            obs, reward, terminated, truncated, info = self.env.step(action)
            while not terminated and not truncated and self.env.current_player != self.controlled_team:
                _, _, terminated, truncated, _ = self.env.step(0)
            return obs, reward, terminated, truncated, info
        else:
            return self.env.step(0)

# Cargar modelos de ciclo 10 y 30
blue_model = A2C.load("backups/a2c_blue_cycle30.zip")
red_model = A2C.load("backups/a2c_red_cycle30.zip")

n_episodes = 100
blue_wins = 0
red_wins = 0
empates = 0

for episode in range(n_episodes):
    env = StrategyEnv5x5Detailed()
    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        obs = env._get_obs()
        if env.current_player == 0:
            action, _ = blue_model.predict(obs, deterministic=True)
        else:
            action, _ = red_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    # Verificar ganador
    alive_blue = any(u.is_alive() and u.team == 0 for u in env.units)
    alive_red = any(u.is_alive() and u.team == 1 for u in env.units)

    if alive_blue and not alive_red:
        blue_wins += 1
    elif alive_red and not alive_blue:
        red_wins += 1
    else:
        empates += 1

print("\nResultados tras 100 partidas:")
print(f"Equipo Azul (Ciclo 10) gana: {blue_wins}")
print(f"Equipo Rojo (Ciclo 30) gana: {red_wins}")
print(f"Empates: {empates}")
