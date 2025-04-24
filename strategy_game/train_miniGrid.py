import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnv1vsDummyMiniGrid import StrategyEnv1vsDummyMiniGrid

# Log para mostrar información cada X pasos
class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            rewards = self.locals.get("rewards")
            if rewards is not None:
                print(f"📈 Paso: {self.n_calls}, Recompensa actual: {rewards}")
        return True

def make_env():
    def _init():
        return StrategyEnv1vsDummyMiniGrid(mode="fixed")
    return _init

if __name__ == "__main__":
    freeze_support()

    model_path = "ppo_minigrid_mimodelofijo"  # ⭐ NOMBRE DEFINIDO AQUÍ

    print("🚀 Entrenando en modo 'fixed'")
    env = DummyVecEnv([make_env()])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=0.01,  # Forzar exploración
    )

    model.learn(total_timesteps=300_000, callback=LogCallback(log_every=5000))
    model.save(model_path)
    print(f"✅ Modelo guardado como: {model_path}.zip")
    env.close()
