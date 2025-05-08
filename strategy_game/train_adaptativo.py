import os
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvBandos import StrategyEnvBandos
from gym_strategy.core.Unit import Soldier, Archer
import gymnasium as gym

# Callback para contar victorias
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
                    print(f"🏆 Victorias - Azul: {self.win_blue} | Rojo: {self.win_red}")
        return True

# Acción máscara
def mask_fn(env):
    return env.unwrapped._get_action_mask()

# Wrapper para filtrar recompensa según equipo
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
        return self.env._get_action_mask()

# Equipos fijos
blue_team = [Soldier, Soldier, Archer]
red_team = [Archer, Soldier, Soldier]

# Entrenamiento adaptativo por self-play
cycles = 10
for i in range(cycles):
    print(f"\n🔁 Ciclo {i+1} - Self-play adaptativo")

    # Cargar modelos previos si existen
    if i == 0:
        red_opponent = None
    else:
        red_opponent = MaskablePPO.load(f"ppo_selfplay_RED_ciclo{i}")

    def make_env_blue():
        env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
        if red_opponent:
            env.red_agent = red_opponent  # oponente fijo para el azul
        return ActionMasker(StrategyWrapper(env, team_controlled=0), mask_fn)

    model_blue = MaskablePPO(
        policy="MultiInputPolicy",
        env=make_env_blue(),
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    )

    model_blue.learn(total_timesteps=100_000, callback=LogCallback())
    model_blue.save(f"ppo_selfplay_BLUE_ciclo{i+1}")

    # Ahora entrenamos el rojo contra el azul recién guardado
    blue_opponent = MaskablePPO.load(f"ppo_selfplay_BLUE_ciclo{i+1}")

    def make_env_red():
        env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
        env.blue_agent = blue_opponent  # oponente fijo para el rojo
        return ActionMasker(StrategyWrapper(env, team_controlled=1), mask_fn)

    model_red = MaskablePPO(
        policy="MultiInputPolicy",
        env=make_env_red(),
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    )

    model_red.learn(total_timesteps=100_000, callback=LogCallback())
    model_red.save(f"ppo_selfplay_RED_ciclo{i+1}")

print("✅ Entrenamiento self-play completado")
