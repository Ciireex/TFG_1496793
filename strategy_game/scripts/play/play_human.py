import os
import sys
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib.ppo_mask import MaskablePPO

if hasattr(sys, '_MEIPASS'):
    print("MEIPASS activo:", sys._MEIPASS)

# === CONFIG ===
HUMAN_TEAM = 0
MODEL_PATH = "maskableppo_red_f7_v3.zip"

# === IMPORTS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === BASE PATH ===
BASE_PATH = getattr(sys, '_MEIPASS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
SPRITE_DIR = os.path.join(BASE_PATH, "sprites")
MODEL_DIR = os.path.join(BASE_PATH, "models")

# === CARGAR MODELO ===
def load_model(path):
    if "maskableppo" in path.lower():
        return MaskablePPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "ppo" in path.lower():
        return PPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "a2c" in path.lower():
        return A2C.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "dqn" in path.lower():
        return DQN.load(path, device="auto")
    else:
        raise ValueError("Modelo no reconocido")

model = load_model(os.path.join(MODEL_DIR, MODEL_PATH))

# === ENTORNO Y RENDERER ===
env = Env_Fase7_Terreno()
obs, _ = env.reset()
done = False

tile_size = 100
cols, rows = env.board_size[1], env.board_size[0]
width = int(tile_size * cols * 1.3)
height = int(tile_size * rows * 0.9)
renderer = Renderer(width=width, height=height, board_size=env.board_size)

# === SPRITES ===
attack_frames = {}

def cargar_sprite_seguro(filename):
    ruta = os.path.join(SPRITE_DIR, filename)
    if os.path.exists(ruta):
        return pygame.image.load(ruta).convert_alpha()
    return None

def cargar_sprites():
    # Soldier Blue
    sprite = cargar_sprite_seguro("SoldierBlue1.png")
    if sprite:
        renderer.override_sprite("Soldier_Team0", sprite)
        attack_frames[(0, "Soldier")] = [cargar_sprite_seguro(f"SoldierBlue{i}.png") for i in range(1, 7)]

    # Soldier Red
    sprite = cargar_sprite_seguro("SoldierRed1.png")
    if sprite:
        renderer.override_sprite("Soldier_Team1", sprite)
        attack_frames[(1, "Soldier")] = [cargar_sprite_seguro(f"SoldierRed{i}.png") for i in range(1, 7)]

    # Archer Blue
    sprite = cargar_sprite_seguro("ArcherBlue1.png")
    if sprite:
        renderer.override_sprite("Archer_Team0", sprite)
        attack_frames[(0, "Archer")] = [cargar_sprite_seguro(f"ArcherBlue{i}.png") for i in range(1, 9)]

    # Archer Red
    sprite = cargar_sprite_seguro("ArcherRed1.png")
    if sprite:
        renderer.override_sprite("Archer_Team1", sprite)
        attack_frames[(1, "Archer")] = [cargar_sprite_seguro(f"ArcherRed{i}.png") for i in range(1, 9)]

    # Knight Blue
    sprite = cargar_sprite_seguro("KnightBlue1.png")
    if sprite:
        renderer.override_sprite("Knight_Team0", sprite)
        attack_frames[(0, "Knight")] = [cargar_sprite_seguro(f"KnightBlue{i}.png") for i in range(1, 7)]

    # Knight Red
    sprite = cargar_sprite_seguro("KnightRed1.png")
    if sprite:
        renderer.override_sprite("Knight_Team1", sprite)
        attack_frames[(1, "Knight")] = [cargar_sprite_seguro(f"KnightRed{i}.png") for i in range(1, 9)]

cargar_sprites()

# === RENDER INICIAL ===
blocked = (env.terrain == 99).astype(np.int8)
renderer.draw_board(units=env.units, terrain=env.terrain,
                    blocked_positions=blocked, active_unit=env._get_active_unit())

clock = pygame.time.Clock()

# === INPUT HUMANO ===
print("Controles:")
print("W = ↑  S = ↓  A = ←  D = →")
print("Q = pasar turno")

KEY_TO_ACTION = {
    pygame.K_w: 3,
    pygame.K_s: 4,
    pygame.K_a: 1,
    pygame.K_d: 2,
    pygame.K_q: 0
}

def get_human_action():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F12:
                    pygame.image.save(renderer.screen, "captura.png")
                    print("🖼️ Pantallazo guardado como 'captura.png'")
                elif event.key in KEY_TO_ACTION:
                    return KEY_TO_ACTION[event.key]

def main():
    global obs, done, blocked

    while not done:
        active_unit = env._get_active_unit()
        team = env.current_player
        action = None

        if team == HUMAN_TEAM:
            print(f"[HUMANO] Fase: {env.phase.upper()}  Unidad: {active_unit.unit_type} Pos: {active_unit.position}")
            action = get_human_action()
        else:
            if isinstance(model, MaskablePPO):
                mask = env.get_action_mask()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model.predict(obs, deterministic=True)

            action = int(action) if not isinstance(action, (np.ndarray, list)) else int(np.asarray(action).item())

        is_attack_phase = env.phase == "attack"
        is_attack_action = action in [1, 2, 3, 4]
        if is_attack_phase and is_attack_action:
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = directions[action - 1]
            tx, ty = active_unit.position[0] + dx, active_unit.position[1] + dy
            target = env._get_unit_at((tx, ty))

            print(f"[CHECK] Intento de ataque hacia ({dx}, {dy}) → Posición destino: ({tx}, {ty})")
            if target:
                print(f"[CHECK] Encontrado objetivo en ({tx}, {ty}): {target.unit_type}, team {target.team}")
            else:
                print(f"[CHECK] No hay objetivo en ({tx}, {ty})")

            key = (active_unit.team, active_unit.unit_type)
            frames = attack_frames.get(key)

            if active_unit.team == 0:
                flip = (dy == -1)
            else:
                flip = (dy == 1)

            if frames:
                print(f"[ANIM] {active_unit.unit_type} del equipo {active_unit.team} ataca hacia ({dx}, {dy})")
                renderer.animate_attack(active_unit, frames, env.terrain, blocked_positions=blocked, target_position=(tx, ty), flip=flip)

        obs, _, done, _, _ = env.step(action)
        blocked = (env.terrain == 99).astype(np.int8)
        renderer.draw_board(units=env.units, terrain=env.terrain,
                            blocked_positions=blocked, active_unit=env._get_active_unit())
        clock.tick(2)

# === EJECUCIÓN PRINCIPAL CON CAPTURA DE ERRORES ===
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pygame.init()
        screen = pygame.display.set_mode((600, 200))
        pygame.display.set_caption("❌ Error al ejecutar")
        font = pygame.font.SysFont(None, 24)
        screen.fill((0, 0, 0))

        error_msg = f"{type(e).__name__}: {str(e)}"
        print(error_msg)

        lines = [error_msg[i:i+70] for i in range(0, len(error_msg), 70)]
        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 0, 0))
            screen.blit(text, (20, 40 + i*30))

        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
