import time
import pygame
from stable_baselines3 import A2C
from gym_strategy.envs.StrategyEnvA2CSplit2v2 import StrategyEnvA2CSplit2v2
from gym_strategy.core.Renderer import Renderer

class StrategyEvalWrapper:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

if __name__ == "__main__":
    model = A2C.load("models/a2c_split2v2")
    env = StrategyEnvA2CSplit2v2()
    env_base = env.unwrapped
    renderer = Renderer(width=60 * 6, height=60 * 4, board_size=env_base.board_size)
    agent = StrategyEvalWrapper(env, model)

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

    move_names = ["quieto", "↑", "↓", "←", "→"]
    done = False
    while not done:
        current_team = env_base.current_player
        team_str = "AZUL" if current_team == 0 else "ROJO"
        units = [u for u in env_base.units if u.team == current_team and u.is_alive()]
        if env_base.active_unit_index < len(units):
            unit = units[env_base.active_unit_index]
        else:
            break

        print(f"\n[{team_str}] Unidad {env_base.active_unit_index + 1} ({unit.unit_type}) en {unit.position}")

        if current_team == 0:
            action = agent.get_action(obs)
        else:
            action = 0  # heurística se ejecuta internamente en el entorno

        if env_base.phase == "move":
            print(f" - Fase: mover → acción: {move_names[action]}")
        else:
            atk_dirs = ["↑", "↓", "←", "→"]
            atk_str = atk_dirs[action - 1] if action > 0 else "pasar"
            print(f" - Fase: atacar → acción: {atk_str}")

        obs, reward, done, _, info = env.step(action)

        renderer.draw_board(
            env_base.units,
            blocked_positions=env_base.blocked_positions,
            capture_point=env_base.capture_point,
            capture_progress=env_base.capture_progress,
            capture_max=env_base.capture_turns_required,
            capturing_team=env_base.current_player
        )

        time.sleep(0.4)

    print("\nPartida terminada. Resultado:", info.get("episode"))
    survivors_blue = sum(1 for u in env_base.units if u.team == 0 and u.is_alive())
    survivors_red = sum(1 for u in env_base.units if u.team == 1 and u.is_alive())
    print(f"Supervivientes - Azul: {survivors_blue}, Rojo: {survivors_red}")
