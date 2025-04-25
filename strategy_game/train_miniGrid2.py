import os
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvMiniGrid2 import StrategyEnv1vsDummyRandomSpawn

# Log para ver el progreso cada X pasos
class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            reward = self.locals.get("rewards")
            ep_info = self.locals.get("infos", [{}])[0].get("episode", None)
            print(f"📈 Paso: {self.n_calls}, Recompensa actual: {reward}")
            if ep_info:
                print(f"   └─ 🎮 Episodio terminado → recompensa total: {ep_info['r']}, longitud: {ep_info['l']}")
        return True

if __name__ == "__main__":
    freeze_support()
    print("🚀 Entrenando con spawns aleatorios y mapas dinámicos")

    env = DummyVecEnv([lambda: StrategyEnv1vsDummyRandomSpawn()])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=1_000_000, callback=LogCallback(log_every=5000))
    model.save("ppo_spawn_v1")
    print("✅ Modelo guardado como 'ppo_spawn_v1.zip'")
