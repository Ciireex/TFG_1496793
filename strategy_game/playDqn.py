import time
import pygame
from stable_baselines3 import DQN
from gym_strategy.envs.StrategyEnvDQNUnitTurn import StrategyEnvDQNUnitTurn
from gym_strategy.core.Renderer import Renderer

class StrategyEvalWrapper:
    def __init__(self, env, model, team_controlled):
        self.env = env
        self.model = model
        self.team = team_controlled

    def get_action(self, obs):
        if self.env.unwrapped.current_player == self.team:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        return 0

if __name__ == "__main__":
    blue_model = DQN.load("models/dqn_blue_cycle5")
    red_model = DQN.load("models/dqn_red_cycle5")

    env = StrategyEnvDQNUnitTurn(team_controlled=0, opponent_model=red_model)
    env_base = env.unwrapped

    renderer = Renderer(width=60 * 9, height=60 * 6, board_size=env_base.board_size)
    agent_blue = StrategyEvalWrapper(env, blue_model, team_controlled=0)
    agent_red = StrategyEvalWrapper(env, red_model, team_controlled=1)

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
        current_team = env_base.current_player
        agent = agent_blue if current_team == 0 else agent_red

        units = [u for u in env_base.units if u.team == current_team and u.is_alive()]
        if env_base.active_unit_index < len(units):
            unit = units[env_base.active_unit_index]
        else:
            break

        action = agent.get_action(obs)
        move_id = action // 5
        atk_id = action % 5

        move_str = ["quieto", "↑", "↓", "←", "→"][move_id]
        atk_str = ["pasar", "↑", "↓", "←", "→"][atk_id]

        print(f"\nEquipo {'Azul' if current_team == 0 else 'Rojo'} - Unidad {env_base.active_unit_index + 1} ({unit.unit_type})")
        print(f" - Posición: {unit.position} | Movimiento: {move_str} | Ataque: {atk_str}")

        obs, reward, done, _, info = env.step(action)

        renderer.draw_board(
            env_base.units,
            blocked_positions=env_base.blocked_positions,
            capture_point=env_base.capture_point,
            capture_progress=env_base.capture_progress,
            capture_max=env_base.capture_turns_required,
            capturing_team=env_base.current_player
        )
        time.sleep(0.3)

    print("\nPartida terminada. Resultado:", info.get("episode"))
    survivors_blue = sum(1 for u in env_base.units if u.team == 0 and u.is_alive())
    survivors_red = sum(1 for u in env_base.units if u.team == 1 and u.is_alive())
    print(f"Supervivientes - Azul: {survivors_blue}, Rojo: {survivors_red}")