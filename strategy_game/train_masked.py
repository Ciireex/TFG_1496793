import os
from multiprocessing import freeze_support
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnvCaptureMaskedDiscrete import StrategyEnvCaptureMaskedDiscrete

def mask_fn(env):
    return env._get_action_mask()

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

    # 1) Crear el entorno base y envolverlo con máscara
    base_env = StrategyEnvCaptureMaskedDiscrete()
    masked_env = ActionMasker(base_env, mask_fn)
    venv = DummyVecEnv([lambda: masked_env])

    # 2) Comprobar si existe un modelo anterior para continuar
    old_model_path = "ppo_capture_masked_v10.zip"
    if os.path.exists(old_model_path):
        print(f"🔄 Cargando modelo anterior '{old_model_path}' para continuar entrenamiento...")
        model = MaskablePPO.load(old_model_path, env=venv)
    else:
        print("🆕 No se encontró modelo anterior. Creando un modelo nuevo desde cero...")
        model = MaskablePPO(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            ent_coef=0.01,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=256,
            clip_range=0.2,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            ),
        )

    # 3) Entrenar el modelo
    model.learn(
        total_timesteps=1_000_000,  # Puedes cambiarlo si quieres
        callback=LogCallback(log_every=5000),
    )

    # 4) Guardar el modelo con NOMBRE NUEVO
    new_model_path = "ppo_capture_masked_v11"
    model.save(new_model_path)
    print(f"✅ Modelo guardado como '{new_model_path}.zip'")
