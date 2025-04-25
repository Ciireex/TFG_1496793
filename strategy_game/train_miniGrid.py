import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnv1vsDummyRandom import StrategyEnv1vsDummyMiniGrid

# Log cada X pasos para ver progreso
class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            rewards = self.locals.get("rewards")
            print(f"📈 Paso: {self.n_calls}, Recompensa actual: {rewards}")
        return True

if __name__ == "__main__":
    freeze_support()

    prev_model_path = "ppo_minigrid_mimodelofijo"  # Modelo entrenado en modo 'fixed'
    new_model_path = "ppo_minigrid_fase2"          # Modelo a guardar en 'preset'

    print("🚀 Continuando entrenamiento en modo 'preset' (fase 2)")

    # ✅ Creamos el entorno vectorizado explícitamente con DummyVecEnv (aunque solo haya 1)
    env = DummyVecEnv([lambda: StrategyEnv1vsDummyMiniGrid(mode="preset")])

    # ✅ Cargamos el modelo anterior y le damos el nuevo entorno
    model = PPO.load(prev_model_path, env=env)

    # ✅ Entrenamiento
    model.learn(total_timesteps=300_000, callback=LogCallback(log_every=5000))

    # ✅ Guardamos
    model.save(new_model_path)
    print(f"✅ Modelo guardado como: {new_model_path}.zip")
