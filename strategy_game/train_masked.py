import os
from multiprocessing import freeze_support

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from gym_strategy.envs.StrategyEnvCaptureMaskedDiscrete import StrategyEnvCaptureMaskedDiscrete

def mask_fn(env):
    return env._get_obs()["action_mask"]

class LogCallback(BaseCallback):
    def __init__(self, log_every=5000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            ep = infos[0].get("episode")
            if ep and self.n_calls % self.log_every == 0:
                r, l = ep["r"], ep["l"]
                print(f"📈 Paso: {self.n_calls}, recompensa: {r:.2f}, longitud: {l}")
        return True

if __name__ == "__main__":
    freeze_support()

    # 1) Entorno base + máscara
    base_env = StrategyEnvCaptureMaskedDiscrete()
    masked = ActionMasker(base_env, mask_fn)
    # 2) VecNormalize para obs y reward
    venv = DummyVecEnv([lambda: masked])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10)

    # 3) Instanciar MaskablePPO con hiperparámetros ajustados
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=venv,
        verbose=1,
        ent_coef=0.05,            # más exploración
        learning_rate=1e-4,       # learning rate más bajo
        n_steps=4096,             # rollout largo
        batch_size=128,           # batch grande
        clip_range=0.1,           # clip estricto
        policy_kwargs=dict(
            net_arch=[dict(pi=[64,64], vf=[64,64])]
        ),
    )

    # 4) Entrenar y guardar
    model.learn(
        total_timesteps=1_000_000,
        callback=LogCallback(log_every=5000),
    )
    model.save("ppo_capture_masked_v4")
    print("✅ Modelo guardado como 'ppo_capture_masked_v4.zip'")
