import os
import sys
from collections import Counter
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from gym_strategy.envs.StrategyEnv_Base import StrategyEnv_2v2Soldiers4x4

# Clase que enfrenta a ambos modelos en turnos reales
class DualPolicyEnv(StrategyEnv_2v2Soldiers4x4):
    def __init__(self, model_blue, model_red):
        super().__init__()
        self.model_blue = model_blue
        self.model_red = model_red

    def step(self, action):
        obs = self._get_obs()
        if self.current_player == 0:
            action, _ = self.model_blue.predict(obs, deterministic=True)
        else:
            action, _ = self.model_red.predict(obs, deterministic=True)
        return super().step(action)

def evaluate(num_episodes=100):
    print("🧠 Cargando modelos...")
    model_blue = PPO.load("../models/ppo_blue_vs_heuristicred/ppo_blue_vf0.zip")
    model_red = PPO.load("../models/ppo_red_vs_frozenblue/ppo_red_vf0.zip")

    env = DualPolicyEnv(model_blue=model_blue, model_red=model_red)

    results = Counter()

    for i in range(num_episodes):
        obs, _ = env.reset(seed=i)
        done = False

        while not done:
            obs = env._get_obs()
            obs, reward, terminated, truncated, _ = env.step(0)
            done = terminated or truncated

        # Determinar ganador según unidades vivas
        blue_alive = sum(1 for u in env.units if u.team == 0 and u.is_alive())
        red_alive = sum(1 for u in env.units if u.team == 1 and u.is_alive())

        if blue_alive > 0 and red_alive == 0:
            results['victorias azules'] += 1
        elif red_alive > 0 and blue_alive == 0:
            results['victorias rojas'] += 1
        else:
            results['empates'] += 1

        print(f"🏁 Partida {i+1}: Azul {blue_alive} vivos, Rojo {red_alive} vivos")

    print("\n📊 RESULTADOS FINALES:")
    print(f"🔵 Victorias azules: {results['victorias azules']}")
    print(f"🔴 Victorias rojas: {results['victorias rojas']}")
    print(f"⚪ Empates: {results['empates']}")

if __name__ == "__main__":
    evaluate(num_episodes=100)
