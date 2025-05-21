import os
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvDef import StrategyEnvDef
import gymnasium as gym

# 📋 Callback para logging por consola
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

# 🧠 Función para extraer la máscara del entorno
def mask_fn(env):
    return env.unwrapped.get_action_mask()

# 🔁 Wrapper para controlar solo un equipo
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

    def get_action_mask(self):
        return self.env.get_action_mask()

# 🧪 Crear entornos para cada equipo
def make_env_blue():
    env = StrategyEnvDef()
    return ActionMasker(StrategyWrapper(env, team_controlled=0), mask_fn)

def make_env_red():
    env = StrategyEnvDef()
    return ActionMasker(StrategyWrapper(env, team_controlled=1), mask_fn)

# 🤖 Crear modelos MaskablePPO con MlpPolicy
model_blue = MaskablePPO(
    policy="MlpPolicy",
    env=make_env_blue(),
    verbose=1,
    learning_rate=1e-4,
    ent_coef=0.005,
    n_steps=2048,
    batch_size=256,
    clip_range=0.2,
)

model_red = MaskablePPO(
    policy="MlpPolicy",
    env=make_env_red(),
    verbose=1,
    learning_rate=1e-4,
    ent_coef=0.005,
    n_steps=2048,
    batch_size=256,
    clip_range=0.2,
)

# 🎯 Entrenamiento por ciclos
os.makedirs("models", exist_ok=True)

for i in range(10):
    print(f"\n🔵 CICLO {i+1} - Entrenando equipo AZUL")
    model_blue.learn(total_timesteps=100_000, callback=LogCallback())
    model_blue.save(f"models/maskppo_v2_blue_cycle{i+1}.zip")

    print(f"\n🔴 CICLO {i+1} - Entrenando equipo ROJO")
    model_red.learn(total_timesteps=100_000, callback=LogCallback())
    model_red.save(f"models/maskppo_v2_red_cycle{i+1}.zip")

print("\n✅ ENTRENAMIENTO MASKABLE PPO V2 COMPLETADO")
