import time
import pygame
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_strategy.envs.StrategyEnvBandos import StrategyEnvBandos
from gym_strategy.core.Unit import Soldier, Archer
from gym_strategy.core.Renderer import Renderer

# Acción máscara
def mask_fn(env):
    return env.unwrapped._get_action_mask()

# Wrapper para evaluación por equipo
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

    env = StrategyEnvBandos(blue_team=blue_team, red_team=red_team)
    env = ActionMasker(env, mask_fn)
    env_base = env.unwrapped

    renderer = Renderer(width=60 * 9, height=60 * 6, board_size=env_base.board_size)

    model_blue = MaskablePPO.load("ppo_bandos_BLUE_ciclo10")
    model_red = MaskablePPO.load("ppo_bandos_RED_ciclo10")

    blue_agent = StrategyEvalWrapper(env, model_blue, team_controlled=0)
    red_agent = StrategyEvalWrapper(env, model_red, team_controlled=1)

    obs, _ = env.reset()
    renderer.draw_board(
        env_base.units,
        blocked_positions=env_base.blocked_positions,
        capture_point=env_base.capture_point,
        capture_progress=env_base.capture_progress,
        capture_max=env_base.capture_turns_required,
        capturing_team=env_base.current_player
    )
    time.sleep(1)

    done = False
    while not done:
        current_team = "Azul" if env_base.current_player == 0 else "Rojo"
        current_agent = blue_agent if env_base.current_player == 0 else red_agent
        actions = current_agent.get_actions(obs)

        print(f"\n🧠 Turno del equipo {current_team}")
        units = [u for u in env_base.units if u.team == env_base.current_player and u.is_alive()]
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attack_dirs = ["arriba", "abajo", "izquierda", "derecha"]

        for unit, action in zip(units, actions):
            if action <= 4:
                move_str = ["quedarse", "arriba", "abajo", "izquierda", "derecha"][action]
                print(f" - {unit.unit_type} en {unit.position} se mueve: {move_str}")
            else:
                dx, dy = dirs[action - 5 + 1]
                direction = attack_dirs[action - 5]
                print(f" - {unit.unit_type} en {unit.position} ataca hacia {direction}")

        obs, reward, done, _, info = env.step(actions)

        renderer.draw_board(
            env_base.units,
            blocked_positions=env_base.blocked_positions,
            capture_point=env_base.capture_point,
            capture_progress=env_base.capture_progress,
            capture_max=env_base.capture_turns_required,
            capturing_team=env_base.current_player
        )
        time.sleep(0.5)

    print("\n✅ Partida terminada. Resultado:", info.get("episode"))
    survivors_blue = sum(1 for u in env_base.units if u.team == 0 and u.is_alive())
    survivors_red = sum(1 for u in env_base.units if u.team == 1 and u.is_alive())
    print(f"Supervivientes - Azul: {survivors_blue}, Rojo: {survivors_red}")
