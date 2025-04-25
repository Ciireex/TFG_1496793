import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvCaptureSparse import StrategyEnvCaptureSparse

# Log cada X pasos
class LogCallback(BaseCallback):
    def __init__(self, log_every=10000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            reward = self.locals.get("rewards")
            ep_info = self.locals.get("infos", [{}])[0].get("episode", None)
            print(f"📈 Paso: {self.n_calls}, Recompensa actual: {reward}")
            if ep_info:
                print(f"   └─ 🎮 Episodio → recompensa: {ep_info['r']}, longitud: {ep_info['l']}")
        return True

if __name__ == "__main__":
    freeze_support()
    print("🚀 Entrenando en entorno escaso (solo recompensa al capturar tras 3 turnos)")

    env = DummyVecEnv([lambda: StrategyEnvCaptureSparse()])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=0.01,         # 🔥 Alta entropía → más exploración
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=1_000_000, callback=LogCallback(log_every=10000))
    model.save("ppo_capture_sparse")
    print("✅ Modelo guardado como 'ppo_capture_sparse.zip'")
