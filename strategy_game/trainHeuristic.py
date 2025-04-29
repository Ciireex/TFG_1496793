import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvAdvance2v2CaptureFocus import StrategyEnvAdvance2v2CaptureFocus

# Función para aplicar la máscara de acciones

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
                    print(f"Paso: {self.n_calls}, recompensa: {r:.2f}, turnos: {l}")
        return True

if __name__ == "__main__":
    freeze_support()

    # 1) Crear entorno
    base_env = StrategyEnvAdvance2v2CaptureFocus()
    masked_env = ActionMasker(base_env, mask_fn)

    # 2) Vectorizar
    venv = DummyVecEnv([lambda: masked_env])

    # 3) Crear modelo
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.001,
        n_steps=4096,
        batch_size=512,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )

    # 4) Entrenar
    model.learn(
        total_timesteps=1_500_000,
        callback=LogCallback(log_every=5000),
    )

    # 5) Guardar
    model.save("ppo_capture_focus")
    print("\u2705 Modelo guardado como 'ppo_capture_focus.zip'")
