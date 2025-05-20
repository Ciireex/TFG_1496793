import pygame
import time
from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPvP import StrategyEnvPvP

# --- Cargar modelo PPO entrenado para el equipo azul (0) ---
model_blue = PPO.load("models/ppo_blue_smallcnn.zip")

# --- Inicializar entorno ---
env = StrategyEnvPvP(render_mode="human")
obs, _ = env.reset()
done = False
clock = pygame.time.Clock()

# --- Función para input manual del jugador rojo (equipo 1) ---
def get_action_from_input():
    move_dir = None
    atk_dir = None
    print("[INPUT] Movimiento (flechas o WASD, espacio para no moverse)...")
    while move_dir is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_SPACE, pygame.K_0, pygame.K_KP0]:
                    move_dir = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    move_dir = 1
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    move_dir = 2
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    move_dir = 3
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    move_dir = 4

    print("[INPUT] Ataque (flechas o WASD, espacio para no atacar)...")
    while atk_dir is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_SPACE, pygame.K_0, pygame.K_KP0]:
                    atk_dir = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    atk_dir = 1
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    atk_dir = 2
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    atk_dir = 3
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    atk_dir = 4

    return [move_dir, atk_dir]

# --- Bucle principal ---
while not done:
    env.render()
    current_team = env.current_player

    if current_team == 0:
        active_unit = env.get_active_unit()
        if active_unit:
            print(f"[IA AZUL] {active_unit.unit_type} en {active_unit.position} con {active_unit.health} HP")
        action, _ = model_blue.predict(obs, deterministic=True)
        print(f"[IA AZUL] Acción: mover={action[0]} | atacar={action[1]}")
        obs, reward, done, truncated, _ = env.step(action)
        time.sleep(1.0)  # Pausa entre acciones para ver lo que hace
    else:
        active_unit = env.get_active_unit()
        if not active_unit:
            continue
        print(f"[TURNO ROJO] {active_unit.unit_type} en {active_unit.position} con {active_unit.health} HP")
        action = get_action_from_input()
        if action is None:
            break
        obs, reward, done, truncated, _ = env.step(action)

    clock.tick(30)

print("Fin de la partida.")
pygame.time.wait(3000)
env.close()
