import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnvDuel2vs2 import StrategyEnvDuel2v2

# Función para aplicar la máscara de acciones válidas
def mask_fn(env):
    return env._get_action_mask()

# Callback para loguear progreso durante el entrenamiento
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

    # 1) Crear el entorno
    base_env = StrategyEnvDuel2v2(training_mode="capture_only")  # 🔵 Primero solo aprenderá a capturar
    masked_env = ActionMasker(base_env, mask_fn)

    # 2) Vectorizar
    venv = DummyVecEnv([lambda: masked_env])

    # 3) Crear el modelo PPO con máscara
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        ent_coef=0.005,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
    )

    # 4) Entrenar el modelo
    model.learn(
        total_timesteps=1_000_000,  # 🚀 Entrenarlo bien para aprender capturar antes de meter combate
        callback=LogCallback(log_every=5000),
    )

    # 5) Guardar el modelo
    model.save("ppo_duel_2v2_v2")
    print("✅ Modelo guardado como 'ppo_duel_2v2_v2.zip'")
