import time
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_strategy.envs.StrategyEnvBandos import StrategyEnvBandos
from gym_strategy.core.Unit import Soldier, Archer

# Acción máscara
def mask_fn(env):
    return env.unwrapped._get_action_mask()

# Wrapper para evaluación de un equipo
class StrategyEvalWrapper:
    def __init__(self, env, model, team_controlled):
        self.env = env
        self.model = model
        self.team = team_controlled

    def get_actions(self, obs):
        if self.env.unwrapped.current_player == self.team:
            action_masks = self.env.unwrapped._get_action_mask()
            action, _ = self.model.predict(obs, action_masks=action_masks)
            return action
        else:
            return [0] * len([
                u for u in self.env.unwrapped.units
                if u.team == self.env.unwrapped.current_player and u.is_alive()
            ])

if __name__ == "__main__":
    blue_team = [Soldier, Soldier, Archer]
    red_team = [Archer, Soldier, Soldier]

    model_blue = MaskablePPO.load("ppo_bandos_BLUE_ciclo10")
    model_red = MaskablePPO.load("ppo_bandos_RED_ciclo10")

    blue_wins = 0
    red_wins = 0
    draws = 0
    total_games = 100

    for i in range(total_games):
        env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
        env = ActionMasker(env, mask_fn)
        env_base = env.unwrapped

        blue_agent = StrategyEvalWrapper(env, model_blue, team_controlled=0)
        red_agent = StrategyEvalWrapper(env, model_red, team_controlled=1)

        obs, _ = env.reset()
        done = False

        while not done:
            current_agent = blue_agent if env_base.current_player == 0 else red_agent
            actions = current_agent.get_actions(obs)
            obs, _, done, _, info = env.step(actions)

        winner = info.get("episode", {}).get("winner", -1)
        if winner == 0:
            blue_wins += 1
        elif winner == 1:
            red_wins += 1
        else:
            draws += 1

        print(f"Partida {i+1}/{total_games} terminada. Ganador: {'Azul' if winner == 0 else 'Rojo' if winner == 1 else 'Empate'}")

    print("\n✅ Evaluación completada")
    print(f"Total partidas: {total_games}")
    print(f"🏆 Victorias Azul: {blue_wins} ({(blue_wins/total_games)*100:.1f}%)")
    print(f"🔴 Victorias Rojo: {red_wins} ({(red_wins/total_games)*100:.1f}%)")
    print(f"➖ Empates: {draws} ({(draws/total_games)*100:.1f}%)")