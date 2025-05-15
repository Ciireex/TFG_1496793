import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvA2CSplit2v2 import StrategyEnvA2CSplit2v2

class LogCallback(BaseCallback):
    def __init__(self, log_every=1000, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every
        self.win_blue = 0
        self.win_red = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                if ep.get("winner") == 0:
                    self.win_blue += 1
                elif ep.get("winner") == 1:
                    self.win_red += 1
                if self.n_calls % self.log_every == 0:
                    print(f"Paso {self.n_calls} | Recompensa: {ep['r']:.2f} | Longitud: {ep['l']} | Wins Azul: {self.win_blue} | Rojo: {self.win_red}")
        return True

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    env = StrategyEnvA2CSplit2v2()

    model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs_a2c_split2v2",
        learning_rate=5e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=True,
        device="auto"
    )

    model.learn(total_timesteps=500_000, callback=LogCallback())
    model.save("models/a2c_split2v2")
