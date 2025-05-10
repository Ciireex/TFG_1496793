import time
import pygame
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_strategy.envs.StrategyEnvTurnBased import StrategyEnvTurnBased
from gym_strategy.core.Unit import Soldier, Archer
from gym_strategy.core.Renderer import Renderer

# Acción máscara
def mask_fn(env):
    return env.unwrapped._get_action_mask()

# Wrapper de evaluación para entornos turn-based por unidad
class StrategyEvalWrapper:
    def __init__(self, env, model, team_controlled):
        self.env = env
        self.model = model
        self.team = team_controlled

    def get_action(self, obs):
        if self.env.unwrapped.current_player == self.team:
            action_mask = self.env.unwrapped._get_action_mask()
            action, _ = self.model.predict(obs, action_masks=action_mask)
            return action
        else:
            return 0  # acción nula cuando no es su turno

if __name__ == "__main__":
    blue_team = [Soldier, Soldier, Archer]
    red_team = [Archer, Soldier, Soldier]

    env = StrategyEnvTurnBased(blue_team=blue_team, red_team=red_team)
    env = ActionMasker(env, mask_fn)
    env_base = env.unwrapped

    renderer = Renderer(width=60 * 9, height=60 * 6, board_size=env_base.board_size)

    model_blue = MaskablePPO.load("ppo_turnbased_BLUE_ciclo10_v4")
    model_red = MaskablePPO.load("ppo_turnbased_RED_ciclo10_v4")

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
        phase = env_base.phase

        print(f"\nTurno del equipo {current_team} - Unidad {env_base.active_unit_index + 1} ({phase})")
        units = [u for u in env_base.units if u.team == env_base.current_player and u.is_alive()]
        if env_base.active_unit_index < len(units):
            unit = units[env_base.active_unit_index]
        else:
            print("No hay unidad activa disponible.")
            break

        action = current_agent.get_action(obs)

        if phase == "move":
            move_str = ["quedarse", "arriba", "abajo", "izquierda", "derecha"][action]
            print(f" - {unit.unit_type} en {unit.position} se mueve: {move_str}")
        else:
            direction = ["arriba", "abajo", "izquierda", "derecha"]
            dir_str = direction[action] if action < len(direction) else "pasar"
            print(f" - {unit.unit_type} en {unit.position} ataca hacia: {dir_str}")

        obs, reward, done, _, info = env.step(action)

        renderer.draw_board(
            env_base.units,
            blocked_positions=env_base.blocked_positions,
            capture_point=env_base.capture_point,
            capture_progress=env_base.capture_progress,
            capture_max=env_base.capture_turns_required,
            capturing_team=env_base.current_player
        )
        time.sleep(0.5)

    print("\nPartida terminada. Resultado:", info.get("episode"))
    survivors_blue = sum(1 for u in env_base.units if u.team == 0 and u.is_alive())
    survivors_red = sum(1 for u in env_base.units if u.team == 1 and u.is_alive())
    print(f"Supervivientes - Azul: {survivors_blue}, Rojo: {survivors_red}")
