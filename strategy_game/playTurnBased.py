import pygame
import time
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvTurnBased import StrategyEnvTurnBased
from gym_strategy.core.Unit import Soldier, Archer
from gym_strategy.core.Renderer import Renderer

# Equipos fijos
blue_team = [Soldier, Soldier, Archer]
red_team = [Archer, Soldier, Soldier]

# Acción máscara
def mask_fn(env):
    return env.unwrapped._get_action_mask()

# Cargar modelos entrenados
model_blue = MaskablePPO.load("ppo_turnbased_BLUE_ciclo8")
model_red = MaskablePPO.load("ppo_turnbased_RED_ciclo8")

# Crear entorno y renderizador
env = StrategyEnvTurnBased(blue_team=blue_team, red_team=red_team)
env = ActionMasker(env, mask_fn)
renderer = Renderer(board_size=env.unwrapped.board_size)

obs, _ = env.reset()
done = False

# Diccionario de acciones
DIRECTIONS = ["quieto", "↑", "↓", "←", "→", "atacar ↑", "atacar ↓", "atacar ←", "atacar →"]

# Bucle de juego
while not done:
    current_team = env.unwrapped.current_player
    model = model_blue if current_team == 0 else model_red

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    renderer.draw_board(
        env.unwrapped.units,
        blocked_positions=env.unwrapped.blocked_positions,
        capture_point=env.unwrapped.capture_point,
        capture_progress=env.unwrapped.capture_progress,
        capturing_team=env.unwrapped.current_player
    )

    active_unit = [u for u in env.unwrapped.units if u.team == current_team and u.is_alive()][env.unwrapped.active_unit_idx - 1]
    action_str = DIRECTIONS[action] if action < len(DIRECTIONS) else f"acción {action}"
    print(f"Turno del equipo {'Azul' if current_team == 0 else 'Rojo'} | Unidad: {active_unit.unit_type} en {active_unit.position} | Acción: {action_str} | Recompensa: {reward:.2f}")

    time.sleep(0.4)

print("\n🎯 Partida terminada")
if info.get("episode"):
    print(f"🏆 Ganador: {'Azul' if info['episode']['winner'] == 0 else 'Rojo'}")