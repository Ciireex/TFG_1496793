import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from gym_strategy.envs.StrategyEnvDQNUnitTurn import StrategyEnvDQNUnitTurn

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

    env = StrategyEnvDQNUnitTurn(team_controlled=0)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs_dqn_unit_turn_v1",
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        device="auto"
    )

    model.learn(total_timesteps=500_000, callback=LogCallback())
    model.save("models/dqn_unit_turn_v1")
