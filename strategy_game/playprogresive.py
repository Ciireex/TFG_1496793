import pygame
from gym_strategy.envs.StrategyEnvPvP import StrategyEnvPvP

# Controles por jugador
KEY_TO_DIR = {
    pygame.K_UP: 1,
    pygame.K_RIGHT: 2,
    pygame.K_DOWN: 3,
    pygame.K_LEFT: 4,
    pygame.K_w: 1,
    pygame.K_d: 2,
    pygame.K_s: 3,
    pygame.K_a: 4,
    pygame.K_0: 0,
    pygame.K_KP0: 0,
    pygame.K_SPACE: 0
}

def get_action_from_input():
    move_dir = None
    atk_dir = None
    print("[INPUT] Esperando dirección de movimiento (flechas o WASD, espacio para no moverse)...")
    while move_dir is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_DIR:
                    move_dir = KEY_TO_DIR[event.key]
                    break

    print("[INPUT] Esperando dirección de ataque (flechas o WASD, espacio para no atacar)...")
    while atk_dir is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_DIR:
                    atk_dir = KEY_TO_DIR[event.key]
                    break

    return [move_dir, atk_dir]

def main():
    env = StrategyEnvPvP(render_mode="human")
    obs, _ = env.reset()

    done = False
    clock = pygame.time.Clock()

    while not done:
        env.render()
        active_unit = env.get_active_unit()
        if active_unit is None:
            continue
        print(f"[TURNO] Equipo {env.current_player} controla unidad {active_unit.unit_type} en {active_unit.position} con {active_unit.health} HP")

        action = get_action_from_input()
        if action is None:
            break  # cerrar
        obs, reward, done, truncated, info = env.step(action)
        clock.tick(30)

    print("¡Fin de la partida!")
    pygame.time.wait(3000)
    env.close()

if __name__ == "__main__":
    main()
