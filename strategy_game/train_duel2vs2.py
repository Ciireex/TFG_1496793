import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvAdvance2v2 import StrategyEnvAdvance2v2

# Función para aplicar la máscara
def mask_fn(env):
    return env._get_action_mask()

# Callback para ver progreso
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
                    print(f"📈 Paso: {self.n_calls}, recompensa media: {r:.2f}, duración media: {l} turnos")
        return True

if __name__ == "__main__":
    freeze_support()

    # 1) Crear entorno
    base_env = StrategyEnvAdvance2v2()
    masked_env = ActionMasker(base_env, mask_fn)

    # 2) Vectorizar
    venv = DummyVecEnv([lambda: masked_env])

    # 3) Crear modelo PPO
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        ent_coef=0.001,             # 🔵 Muy baja entropía ➔ prioriza la mejor acción
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 🔵 Red decente para 2v2
        ),
    )

    # 4) Entrenar modelo
    model.learn(
        total_timesteps=2_000_000,  # 🚀 Entrenarlo a fondo
        callback=LogCallback(log_every=5000),
    )

    # 5) Guardar modelo
    model.save("ppo_advance2v2_final")
    print("✅ Modelo guardado como 'ppo_advance2v2_final.zip'")
