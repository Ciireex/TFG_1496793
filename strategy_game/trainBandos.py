import os
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnvBandos import StrategyEnvBandos
from gym_strategy.core.Unit import Soldier, Archer
import gymnasium as gym

class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            for info in infos:
                ep = info.get("episode")
                if ep and self.n_calls % self.log_every == 0:
                    r, l = ep["r"], ep["l"]
                    print(f"Paso {self.n_calls} | Recompensa: {r:.2f} | Longitud: {l}")
        return True

def mask_fn(env):
    return env.unwrapped._get_action_mask()

class StrategyWrapper(gym.Wrapper):
    def __init__(self, env, team_controlled):
        super().__init__(env)
        self.team_controlled = team_controlled

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self.env.current_player != self.team_controlled:
            reward = 0.0
        return obs, reward, done, truncated, info

    def get_action_mask(self):
        return self.env._get_action_mask()

if __name__ == "__main__":
    blue_team = [Soldier, Soldier, Archer]
    red_team = [Archer, Soldier, Soldier]

    def make_env_blue():
        env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
        return ActionMasker(StrategyWrapper(env, team_controlled=0), mask_fn)

    def make_env_red():
        env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
        return ActionMasker(StrategyWrapper(env, team_controlled=1), mask_fn)

    env_blue = make_env_blue()
    env_red = make_env_red()

    model_blue = MaskablePPO(
        policy="MultiInputPolicy",
        env=env_blue,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    )

    model_red = MaskablePPO(
        policy="MultiInputPolicy",
        env=env_red,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    )

    for i in range(10):
        print(f"\n🎯 Entrenando equipo AZUL - ciclo {i + 1}")
        model_blue.learn(total_timesteps=100_000, callback=LogCallback(log_every=5000))

        print(f"\n🎯 Entrenando equipo ROJO - ciclo {i + 1}")
        model_red.learn(total_timesteps=100_000, callback=LogCallback(log_every=5000))

    model_blue.save("ppo_bandos_v2_BLUE")
    model_red.save("ppo_bandos_v2_RED")
    print("Modelos guardados: ppo_bandos_v2_BLUE.zip y ppo_bandos_v2_RED.zip")
