import os
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_strategy.envs.Env3v3 import Env3v3  # ← tu nuevo entorno
from gym_strategy.core.Unit import Soldier, Archer

# Cambia esto para elegir algoritmo
USE_MASKABLE = False  # True = MaskablePPO | False = A2C

# Función para aplicar máscara (solo si se usa MaskablePPO)
def mask_fn(env):
    return env._get_action_mask()

# Callback para mostrar progreso
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
                    print(f"📈 Paso {self.n_calls} | Recompensa: {r:.2f} | Longitud: {l}")
        return True

if __name__ == "__main__":
    # Composición de los equipos
    blue_team = [Soldier, Soldier, Archer]
    red_team = [Archer, Soldier, Soldier]

    # Crear entorno
    def make_env():
        env = Env3v3(blue_team=blue_team, red_team=red_team)
        return ActionMasker(env, mask_fn) if USE_MASKABLE else env

    env = DummyVecEnv([make_env])

    # (PPO o A2C)
    if USE_MASKABLE:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            ent_coef=0.005,
            n_steps=4096,
            batch_size=256,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        )
        model_name = "ppo_3v3_soldiers_archers"
    else:
        model = A2C(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=20,
            policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        )
        model_name = "a2c_3v3_soldiers_archers"

    # Entrenamiento
    model.learn(total_timesteps=1_000_000, callback=LogCallback(log_every=5000))

    # Guardar modelo
    model.save(model_name)
    print(f"Modelo guardado como '{model_name}.zip'")
