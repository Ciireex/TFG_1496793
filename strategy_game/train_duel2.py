import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnvDuel import StrategyEnvDuel

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
    base_env = StrategyEnvDuel()
    masked_env = ActionMasker(base_env, mask_fn)

    # 2) Vectorizar
    venv = DummyVecEnv([lambda: masked_env])

    # 3) Crear el modelo PPO con máscara
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        ent_coef=0.005,          # 🔵 Baja entropía para menos exploración aleatoria
        learning_rate=1e-4,      # Estándar
        n_steps=4096,            # N pasos por batch
        batch_size=256,          # Tamaño de batch
        clip_range=0.2,          # PPO clip range
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # 🔥 Red decente para decidir bien ataque vs captura
        ),
    )

    # 4) Entrenar el modelo
    model.learn(
        total_timesteps=2_000_000,  # 🚀 2 millones de pasos para darle tiempo a aprender capturar bien
        callback=LogCallback(log_every=5000),
    )

    # 5) Guardar el modelo
    model.save("ppo_duel_v5")
    print("✅ Modelo guardado como 'ppo_duel_v5.zip'")
