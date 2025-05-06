import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnv2v2 import StrategyEnv2v2

# Función para aplicar la máscara de acciones válidas
def mask_fn(env):
    return env._get_action_mask()

# Callback personalizado para loguear el progreso
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
                    print(f"📈 Paso: {self.n_calls}, recompensa: {r:.2f}, longitud: {l}")
        return True

if __name__ == "__main__":
    freeze_support()

    # Crear entorno con máscara
    base_env = StrategyEnv2v2()
    masked_env = ActionMasker(base_env, mask_fn)

    # Vectorizar
    venv = DummyVecEnv([lambda: masked_env])

    # Crear el modelo con red adecuada
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=venv,
        verbose=1,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=512,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
    )

    # Entrenar
    model.learn(
        total_timesteps=800_000,
        callback=LogCallback(log_every=5000),
    )

    # Guardar modelo
    model.save("ppo_2v2_extremos")
    print("✅ Modelo guardado como 'ppo_2v2_extremos.zip'")
