import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvSoldiersOnly import StrategyEnvSoldiersOnly
from gym_strategy.core.Unit import Soldier
import gymnasium as gym

# Callback para logging
class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every
        self.win_blue = 0
        self.win_red = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                if ep.get("winner") == 0:
                    self.win_blue += 1
                elif ep.get("winner") == 1:
                    self.win_red += 1

                if self.n_calls % self.log_every == 0:
                    print(f"Paso {self.n_calls} | Recompensa: {ep['r']:.2f} | Longitud: {ep['l']}")
                    print(f"Victorias - Azul: {self.win_blue} | Rojo: {self.win_red}")
        return True

# Wrapper para controlar un solo equipo
class StrategyWrapper(gym.Wrapper):
    def __init__(self, env, team_controlled):
        super().__init__(env)
        self.team_controlled = team_controlled

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.current_player != self.team_controlled:
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Crear entornos

def make_env_blue():
    env = StrategyEnvSoldiersOnly()
    return StrategyWrapper(env, team_controlled=0)

def make_env_red():
    env = StrategyEnvSoldiersOnly()
    return StrategyWrapper(env, team_controlled=1)

# Crear modelos
model_blue = A2C(
    policy="MlpPolicy",
    env=make_env_blue(),
    verbose=1,
    tensorboard_log="./a2c_logs",
    learning_rate=1e-4,
    ent_coef=0.005,
    n_steps=2048,
    gamma=0.99
)

model_red = A2C(
    policy="MlpPolicy",
    env=make_env_red(),
    verbose=1,
    tensorboard_log="./a2c_logs",
    learning_rate=1e-4,
    ent_coef=0.005,
    n_steps=2048,
    gamma=0.99
)

# Entrenamiento alternado
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

for i in range(10):
    print(f"\nCiclo {i+1} - Entrenando equipo AZUL")
    model_blue.learn(total_timesteps=100_000, callback=LogCallback(), progress_bar=True)
    model_blue.save(os.path.join(output_dir, f"a2c_BLUE_ciclo{i+1}"))

    print(f"\nCiclo {i+1} - Entrenando equipo ROJO")
    model_red.learn(total_timesteps=100_000, callback=LogCallback(), progress_bar=True)
    model_red.save(os.path.join(output_dir, f"a2c_RED_ciclo{i+1}"))

print("Entrenamiento turn-based con A2C finalizado.")
