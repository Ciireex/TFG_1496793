import time
import pygame
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvDef import StrategyEnvDef
from gym_strategy.core.Renderer import Renderer

# Funcion de enmascarado para MaskablePPO
def mask_fn(env):
    return env.unwrapped.get_action_mask()

# Cargar modelos entrenados
blue_model = MaskablePPO.load("models/maskppo_v2_blue_cycle10.zip")
red_model = MaskablePPO.load("models/maskppo_v2_red_cycle10.zip")

# Inicializar entorno
env = ActionMasker(StrategyEnvDef(), mask_fn)
base_env = env.unwrapped
renderer = Renderer(board_size=base_env.board_size)

obs, _ = env.reset()
terminated = False
truncated = False
DIRECTIONS = ["quieto", "↑", "↓", "←", "→"]

while not terminated and not truncated:
    current_team = base_env.current_player
    phase = base_env.phase
    team_units = [u for u in base_env.units if u.team == current_team and u.is_alive()]
    active_unit = team_units[base_env.active_unit_index] if base_env.active_unit_index < len(team_units) else None

    print(f"\nTurno del equipo {'AZUL' if current_team == 0 else 'ROJO'} - Fase: {phase.upper()}")
    print(f"Progreso de captura - AZUL: {base_env.capture_progress[0]}/3 | ROJO: {base_env.capture_progress[1]}/3")

    renderer.draw_board(
        units=base_env.units,
        blocked_positions=[(x, y) for x in range(base_env.board_size[0]) for y in range(base_env.board_size[1]) if base_env.obstacles[x, y] == 1],
        capture_point=base_env.capture_point,
        capture_progress=base_env.capture_progress,
        capture_max=base_env.capture_turns_required,
        capturing_team=base_env.current_player,
        active_unit=active_unit,
        highlight_attack=True
    )

    model = blue_model if current_team == 0 else red_model
    action_mask = base_env.get_action_mask()
    print("Máscara de acción:", list(action_mask))

    action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
    print("Acción elegida:", DIRECTIONS[action] if action < len(DIRECTIONS) else action)

    obs, reward, terminated, truncated, info = env.step(action)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

# Resultado final
alive_blue = any(u.is_alive() and u.team == 0 for u in base_env.units)
alive_red = any(u.is_alive() and u.team == 1 for u in base_env.units)

print("\n=== RESULTADO FINAL ===")
if base_env.capture_progress[0] >= base_env.capture_turns_required:
    print("Gana el EQUIPO AZUL por captura")
elif base_env.capture_progress[1] >= base_env.capture_turns_required:
    print("Gana el EQUIPO ROJO por captura")
elif alive_blue and not alive_red:
    print("Gana el EQUIPO AZUL por eliminación")
elif alive_red and not alive_blue:
    print("Gana el EQUIPO ROJO por eliminación")
else:
    print("Empate")

print("\nPulsa cualquier tecla para cerrar la ventana...")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            running = False
pygame.quit()